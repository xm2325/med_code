from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from cohortcoder.mipa_local_inference import (
    InferenceConfig,
    generate_local_predictions,
    parse_model_response,
)
from cohortcoder.mipa_phenotyping import AcceptanceThresholds, evaluate_mipa_predictions


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _backend(path: Path, *, malformed: bool = False) -> Path:
    if malformed:
        source = "print('not-json')\n"
    else:
        source = r'''
import json, sys
request = json.loads(sys.stdin.read())
phenotype = request["phenotype"]
user = request["messages"][1]["content"]
if phenotype == "hypertension" and "Hypertension documented." in user:
    response = {
        "prediction": 1,
        "evidence": "Hypertension documented.",
        "assertion": "present",
        "temporality": "current",
        "confidence": 0.99,
    }
else:
    response = {
        "prediction": 0,
        "evidence": "",
        "assertion": "absent",
        "temporality": "unknown",
        "confidence": 0.95,
    }
print(json.dumps(response))
'''
    path.write_text(source, encoding="utf-8")
    return path


def test_strict_parser_rejects_markdown_fences() -> None:
    with pytest.raises(ValueError, match="markdown fences"):
        parse_model_response(
            '```json\n{"prediction": 0, "evidence": "", "assertion": "absent", '
            '"temporality": "unknown", "confidence": 0.9}\n```'
        )


def test_generator_checkpoints_resumes_and_does_not_write_full_note(tmp_path: Path) -> None:
    notes = tmp_path / "notes.csv"
    predictions = tmp_path / "predictions.jsonl"
    failures = tmp_path / "failures.jsonl"
    backend = _backend(tmp_path / "backend.py")
    full_note = "Private synthetic note. Hypertension documented."
    _write_csv(notes, [{"note_id": "n1", "text": full_note}], ["note_id", "text"])

    config = InferenceConfig(command=(sys.executable, str(backend)), model_id="mock-local")
    first = generate_local_predictions(
        notes_path=notes,
        predictions_path=predictions,
        failures_path=failures,
        config=config,
        phenotypes=("hypertension", "depression"),
    )
    assert first["complete"] is True
    assert first["counts"]["generated_this_run"] == 2
    output_text = predictions.read_text(encoding="utf-8")
    assert full_note not in output_text
    rows = [json.loads(line) for line in output_text.splitlines()]
    assert {(row["note_id"], row["phenotype"]) for row in rows} == {
        ("n1", "hypertension"),
        ("n1", "depression"),
    }

    second = generate_local_predictions(
        notes_path=notes,
        predictions_path=predictions,
        failures_path=failures,
        config=config,
        phenotypes=("hypertension", "depression"),
    )
    assert second["counts"]["generated_this_run"] == 0
    assert second["counts"]["skipped_existing"] == 2
    assert len(predictions.read_text(encoding="utf-8").splitlines()) == 2


def test_malformed_backend_is_logged_and_fails_completeness(tmp_path: Path) -> None:
    notes = tmp_path / "notes.csv"
    predictions = tmp_path / "predictions.jsonl"
    failures = tmp_path / "failures.jsonl"
    backend = _backend(tmp_path / "bad_backend.py", malformed=True)
    _write_csv(notes, [{"note_id": "n1", "text": "Synthetic note."}], ["note_id", "text"])

    summary = generate_local_predictions(
        notes_path=notes,
        predictions_path=predictions,
        failures_path=failures,
        config=InferenceConfig(command=(sys.executable, str(backend))),
        phenotypes=("hypertension",),
    )
    assert summary["complete"] is False
    assert summary["counts"]["failures_this_run"] == 1
    failure = json.loads(failures.read_text(encoding="utf-8").strip())
    assert failure["error_type"] == "ValueError"


def test_local_generation_to_final_acceptance_pass(tmp_path: Path) -> None:
    notes = tmp_path / "notes.csv"
    labels = tmp_path / "labels.csv"
    predictions = tmp_path / "predictions.jsonl"
    failures = tmp_path / "failures.jsonl"
    audit = tmp_path / "audit.csv"
    backend = _backend(tmp_path / "backend.py")

    _write_csv(
        notes,
        [
            {"note_id": "n1", "subject_id": "s1", "hadm_id": "h1", "text": "Hypertension documented."},
            {"note_id": "n2", "subject_id": "s2", "hadm_id": "h2", "text": "No target disease documented."},
        ],
        ["note_id", "subject_id", "hadm_id", "text"],
    )
    _write_csv(
        labels,
        [
            {"note_id": "n1", "subject_id": "s1", "hadm_id": "h1", "hypertension": 1},
            {"note_id": "n2", "subject_id": "s2", "hadm_id": "h2", "hypertension": 0},
        ],
        ["note_id", "subject_id", "hadm_id", "hypertension"],
    )
    _write_csv(
        audit,
        [{"note_id": "n1", "phenotype": "hypertension", "supports_prediction": 1, "severe_context_error": 0}],
        ["note_id", "phenotype", "supports_prediction", "severe_context_error"],
    )

    generation = generate_local_predictions(
        notes_path=notes,
        predictions_path=predictions,
        failures_path=failures,
        config=InferenceConfig(command=(sys.executable, str(backend)), model_id="mock-local"),
        phenotypes=("hypertension",),
    )
    assert generation["complete"] is True

    summary, metrics, errors, _ = evaluate_mipa_predictions(
        labels_path=labels,
        notes_path=notes,
        predictions_path=predictions,
        phenotypes=("hypertension",),
        evidence_audit_path=audit,
        thresholds=AcceptanceThresholds(min_phenotypes_passing=1, common_positive_min=1),
    )
    assert metrics[0]["f1"] == 1.0
    assert errors == []
    assert summary["acceptance"]["final_status"] == "PASS"
