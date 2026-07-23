from __future__ import annotations

import csv
from pathlib import Path

from cohortcoder.mipa_phenotyping import (
    DEFAULT_PHENOTYPES,
    AcceptanceThresholds,
    build_subject_split_manifest,
    evaluate_mipa_predictions,
    split_audit,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path, *, wrong: bool = False, with_audit: bool = False):
    labels = []
    notes = []
    predictions = []
    audit = []
    for index, positive_phenotype in enumerate(DEFAULT_PHENOTYPES, start=1):
        note_id = str(index)
        subject_id = str((index + 1) // 2)  # repeated subjects test patient-level grouping
        phrase = f"Evidence for {positive_phenotype}."
        labels.append(
            {
                "note_id": note_id,
                "subject_id": subject_id,
                "hadm_id": str(1000 + index),
                **{phenotype: int(phenotype == positive_phenotype) for phenotype in DEFAULT_PHENOTYPES},
            }
        )
        notes.append(
            {
                "note_id": note_id,
                "subject_id": subject_id,
                "hadm_id": str(1000 + index),
                "text": f"Synthetic local test note. {phrase}",
            }
        )
        for phenotype in DEFAULT_PHENOTYPES:
            prediction = int(phenotype == positive_phenotype)
            if wrong and note_id == "1" and phenotype == positive_phenotype:
                prediction = 0
            predictions.append(
                {
                    "note_id": note_id,
                    "phenotype": phenotype,
                    "prediction": prediction,
                    "evidence": phrase if prediction else "",
                    "assertion": "present" if prediction else "absent",
                    "temporality": "current" if prediction else "unknown",
                }
            )
            if with_audit and prediction:
                audit.append(
                    {
                        "note_id": note_id,
                        "phenotype": phenotype,
                        "supports_prediction": 1,
                        "severe_context_error": 0,
                    }
                )

    labels_path = tmp_path / "labels.csv"
    notes_path = tmp_path / "notes.csv"
    predictions_path = tmp_path / "predictions.csv"
    audit_path = tmp_path / "audit.csv"
    _write_csv(labels_path, labels, ["note_id", "subject_id", "hadm_id", *DEFAULT_PHENOTYPES])
    _write_csv(notes_path, notes, ["note_id", "subject_id", "hadm_id", "text"])
    _write_csv(
        predictions_path,
        predictions,
        ["note_id", "phenotype", "prediction", "evidence", "assertion", "temporality"],
    )
    if with_audit:
        _write_csv(
            audit_path,
            audit,
            ["note_id", "phenotype", "supports_prediction", "severe_context_error"],
        )
    return labels_path, notes_path, predictions_path, audit_path if with_audit else None


def test_subject_split_is_disjoint_with_repeated_subjects(tmp_path: Path) -> None:
    labels_path, _, _, _ = _fixture(tmp_path)
    with labels_path.open("r", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    manifest = build_subject_split_manifest(labels)
    audit = split_audit(manifest)
    assert audit["subject_disjoint"] is True
    subject_splits: dict[str, set[str]] = {}
    for row in manifest:
        subject_splits.setdefault(row["subject_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in subject_splits.values())


def test_perfect_predictions_pass_automated_gate_but_require_human_audit(tmp_path: Path) -> None:
    labels, notes, predictions, _ = _fixture(tmp_path)
    summary, metrics, errors, _ = evaluate_mipa_predictions(
        labels_path=labels,
        notes_path=notes,
        predictions_path=predictions,
        thresholds=AcceptanceThresholds(common_positive_min=1),
    )
    assert summary["performance"]["macro_f1"] == 1.0
    assert summary["acceptance"]["automated_gate_pass"] is True
    assert summary["acceptance"]["final_status"] == "PASS_AUTOMATED_PENDING_HUMAN_EVIDENCE_AUDIT"
    assert summary["evidence_contract"]["assertion_temporality_accuracy_scored"] is False
    assert len(metrics) == len(DEFAULT_PHENOTYPES)
    assert errors == []


def test_human_evidence_audit_can_complete_final_pass(tmp_path: Path) -> None:
    labels, notes, predictions, audit = _fixture(tmp_path, with_audit=True)
    summary, _, _, _ = evaluate_mipa_predictions(
        labels_path=labels,
        notes_path=notes,
        predictions_path=predictions,
        evidence_audit_path=audit,
        thresholds=AcceptanceThresholds(common_positive_min=1),
    )
    assert summary["human_evidence_audit"]["evidence_support_rate"] == 1.0
    assert summary["human_evidence_audit"]["severe_context_error_rate"] == 0.0
    assert summary["acceptance"]["final_status"] == "PASS"


def test_classification_error_fails_automated_gate(tmp_path: Path) -> None:
    labels, notes, predictions, _ = _fixture(tmp_path, wrong=True)
    summary, _, errors, _ = evaluate_mipa_predictions(
        labels_path=labels,
        notes_path=notes,
        predictions_path=predictions,
        thresholds=AcceptanceThresholds(common_positive_min=1),
    )
    assert summary["acceptance"]["automated_gate_pass"] is False
    assert summary["acceptance"]["final_status"] == "FAIL_AUTOMATED_GATE"
    assert any("classification_error" in str(row["error_type"]) for row in errors)
