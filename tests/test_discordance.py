from __future__ import annotations

import csv
import json
from pathlib import Path

from cohortcoder.discordance import evaluate_three_way_discordance
from cohortcoder.mipa_phenotyping import DEFAULT_PHENOTYPES


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path, *, missing_structured: bool = False):
    labels = []
    text = []
    structured = []
    for index, positive in enumerate(DEFAULT_PHENOTYPES, start=1):
        note_id = str(index)
        labels.append(
            {
                "note_id": note_id,
                "subject_id": str(index),
                **{phenotype: int(phenotype == positive) for phenotype in DEFAULT_PHENOTYPES},
            }
        )
        for phenotype in DEFAULT_PHENOTYPES:
            gold = int(phenotype == positive)
            text.append({"note_id": note_id, "phenotype": phenotype, "prediction": gold})
            # Structured data intentionally misses every true phenotype and correctly leaves negatives absent.
            if not (missing_structured and note_id == "1" and phenotype == DEFAULT_PHENOTYPES[0]):
                structured.append({"note_id": note_id, "phenotype": phenotype, "structured_positive": 0})

    labels_path = tmp_path / "labels.csv"
    text_path = tmp_path / "text.csv"
    structured_path = tmp_path / "structured.csv"
    upstream_path = tmp_path / "upstream.json"
    _write_csv(labels_path, labels, ["note_id", "subject_id", *DEFAULT_PHENOTYPES])
    _write_csv(text_path, text, ["note_id", "phenotype", "prediction"])
    _write_csv(structured_path, structured, ["note_id", "phenotype", "structured_positive"])
    upstream_path.write_text(json.dumps({"acceptance": {"final_status": "PASS"}}), encoding="utf-8")
    return labels_path, text_path, structured_path, upstream_path


def test_three_way_metrics_recover_gold_positive_code_absent_cases(tmp_path: Path) -> None:
    labels, text, structured, upstream = _fixture(tmp_path)
    summary, metrics, cells = evaluate_three_way_discordance(
        labels_path=labels,
        text_predictions_path=text,
        structured_status_path=structured,
        upstream_summary_path=upstream,
        structured_scope_validated=True,
    )
    assert summary["coverage"]["complete"] is True
    assert summary["overall"]["gold_positive"] == len(DEFAULT_PHENOTYPES)
    assert summary["overall"]["structured_sensitivity"] == 0.0
    assert summary["overall"]["text_sensitivity"] == 1.0
    assert summary["overall"]["combined_sensitivity"] == 1.0
    assert summary["overall"]["gold_positive_code_absent"] == len(DEFAULT_PHENOTYPES)
    assert summary["overall"]["gold_positive_text_positive_code_absent"] == len(DEFAULT_PHENOTYPES)
    assert summary["overall"]["recoverable_code_missed_fraction"] == 1.0
    assert summary["interpretation"]["status"] == "CONFIRMATORY_ELIGIBLE"
    assert len(metrics) == len(DEFAULT_PHENOTYPES)
    assert len(cells) == len(DEFAULT_PHENOTYPES) * 8


def test_upstream_nonpass_forces_exploratory_interpretation(tmp_path: Path) -> None:
    labels, text, structured, upstream = _fixture(tmp_path)
    upstream.write_text(
        json.dumps({"acceptance": {"final_status": "PASS_AUTOMATED_PENDING_HUMAN_EVIDENCE_AUDIT"}}),
        encoding="utf-8",
    )
    summary, _, _ = evaluate_three_way_discordance(
        labels_path=labels,
        text_predictions_path=text,
        structured_status_path=structured,
        upstream_summary_path=upstream,
        structured_scope_validated=True,
    )
    assert summary["upstream_gate"]["stage1_2_pass"] is False
    assert summary["interpretation"]["confirmatory_underrecording_language_allowed"] is False
    assert summary["interpretation"]["status"] == "EXPLORATORY_DISCORDANCE_ONLY"


def test_unvalidated_structured_scope_forces_exploratory_interpretation(tmp_path: Path) -> None:
    labels, text, structured, upstream = _fixture(tmp_path)
    summary, _, _ = evaluate_three_way_discordance(
        labels_path=labels,
        text_predictions_path=text,
        structured_status_path=structured,
        upstream_summary_path=upstream,
        structured_scope_validated=False,
    )
    assert summary["structured_definition"]["scope_validated"] is False
    assert summary["interpretation"]["status"] == "EXPLORATORY_DISCORDANCE_ONLY"


def test_missing_structured_pair_is_reported_as_incomplete_coverage(tmp_path: Path) -> None:
    labels, text, structured, upstream = _fixture(tmp_path, missing_structured=True)
    summary, _, _ = evaluate_three_way_discordance(
        labels_path=labels,
        text_predictions_path=text,
        structured_status_path=structured,
        upstream_summary_path=upstream,
        structured_scope_validated=True,
    )
    assert summary["coverage"]["text_prediction_coverage"] == 1.0
    assert summary["coverage"]["structured_status_coverage"] < 1.0
    assert summary["coverage"]["complete"] is False
    assert summary["interpretation"]["confirmatory_underrecording_language_allowed"] is False
