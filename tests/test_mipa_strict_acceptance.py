from __future__ import annotations

import csv
import json
from pathlib import Path

from cohortcoder.mipa_phenotyping import AcceptanceThresholds
from cohortcoder.mipa_strict_acceptance import evaluate_mipa_predictions_strict


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path, *, full_audit: bool):
    labels = tmp_path / "labels.csv"
    notes = tmp_path / "notes.csv"
    predictions = tmp_path / "predictions.jsonl"
    audit = tmp_path / "audit.csv"

    _write_csv(
        labels,
        [
            {"note_id": "n1", "subject_id": "s1", "hadm_id": "h1", "hypertension": 1},
            {"note_id": "n2", "subject_id": "s2", "hadm_id": "h2", "hypertension": 1},
        ],
        ["note_id", "subject_id", "hadm_id", "hypertension"],
    )
    _write_csv(
        notes,
        [
            {"note_id": "n1", "subject_id": "s1", "hadm_id": "h1", "text": "Hypertension A."},
            {"note_id": "n2", "subject_id": "s2", "hadm_id": "h2", "text": "Hypertension B."},
        ],
        ["note_id", "subject_id", "hadm_id", "text"],
    )
    rows = [
        {
            "note_id": "n1",
            "phenotype": "hypertension",
            "prediction": 1,
            "evidence": "Hypertension A.",
            "assertion": "present",
            "temporality": "current",
        },
        {
            "note_id": "n2",
            "phenotype": "hypertension",
            "prediction": 1,
            "evidence": "Hypertension B.",
            "assertion": "present",
            "temporality": "current",
        },
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit_rows = [
        {"note_id": "n1", "phenotype": "hypertension", "supports_prediction": 1, "severe_context_error": 0}
    ]
    if full_audit:
        audit_rows.append(
            {"note_id": "n2", "phenotype": "hypertension", "supports_prediction": 1, "severe_context_error": 0}
        )
    _write_csv(
        audit,
        audit_rows,
        ["note_id", "phenotype", "supports_prediction", "severe_context_error"],
    )
    return labels, notes, predictions, audit


def test_partial_positive_audit_cannot_receive_final_pass(tmp_path: Path) -> None:
    labels, notes, predictions, audit = _fixture(tmp_path, full_audit=False)
    summary, _, _, _ = evaluate_mipa_predictions_strict(
        labels_path=labels,
        notes_path=notes,
        predictions_path=predictions,
        phenotypes=("hypertension",),
        evidence_audit_path=audit,
        thresholds=AcceptanceThresholds(min_phenotypes_passing=1, common_positive_min=1),
    )
    assert summary["acceptance"]["automated_gate_pass"] is True
    assert summary["human_evidence_audit"]["positive_prediction_audit_coverage"] == 0.5
    assert summary["acceptance"]["human_gate_checks"]["positive_prediction_audit_coverage"] is False
    assert summary["acceptance"]["final_status"] == "FAIL_HUMAN_EVIDENCE_GATE"


def test_complete_positive_audit_can_receive_final_pass(tmp_path: Path) -> None:
    labels, notes, predictions, audit = _fixture(tmp_path, full_audit=True)
    summary, _, _, _ = evaluate_mipa_predictions_strict(
        labels_path=labels,
        notes_path=notes,
        predictions_path=predictions,
        phenotypes=("hypertension",),
        evidence_audit_path=audit,
        thresholds=AcceptanceThresholds(min_phenotypes_passing=1, common_positive_min=1),
    )
    assert summary["human_evidence_audit"]["positive_prediction_audit_coverage"] == 1.0
    assert summary["acceptance"]["final_status"] == "PASS"
