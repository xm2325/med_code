"""Strict scientific acceptance wrapper for MIPA phenotype evaluation.

The underlying evaluator computes classification and evidence-quality metrics. This wrapper adds
an explicit human-audit coverage gate so a small hand-picked subset cannot produce final PASS.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .mipa_phenotyping import (
    DEFAULT_PHENOTYPES,
    AcceptanceThresholds,
    _binary,
    _read_predictions,
    evaluate_mipa_predictions,
)


@dataclass(frozen=True)
class EvidenceAuditCoveragePolicy:
    min_positive_prediction_coverage: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_positive_prediction_coverage <= 1.0:
            raise ValueError("min_positive_prediction_coverage must be between 0 and 1")


def _read_audit_keys(path: str | Path | None) -> set[tuple[str, str]]:
    if path is None:
        return set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return set()
    required = {"note_id", "phenotype", "supports_prediction", "severe_context_error"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Evidence audit schema missing columns: {sorted(missing)}")
    keys = [(str(row["note_id"]), str(row["phenotype"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Evidence audit contains duplicate note_id/phenotype keys")
    return set(keys)


def evaluate_mipa_predictions_strict(
    *,
    labels_path: str | Path,
    notes_path: str | Path,
    predictions_path: str | Path,
    phenotypes: Sequence[str] = DEFAULT_PHENOTYPES,
    evaluation_scope: str = "all",
    split_seed: str = "20260723",
    evidence_audit_path: str | Path | None = None,
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
    audit_policy: EvidenceAuditCoveragePolicy = EvidenceAuditCoveragePolicy(),
):
    """Run evaluation and require prespecified coverage of all positive predictions for final PASS."""
    summary, metrics, errors, manifest = evaluate_mipa_predictions(
        labels_path=labels_path,
        notes_path=notes_path,
        predictions_path=predictions_path,
        phenotypes=phenotypes,
        evaluation_scope=evaluation_scope,
        split_seed=split_seed,
        evidence_audit_path=evidence_audit_path,
        thresholds=thresholds,
    )

    split_by_note = {str(row["note_id"]): str(row["split"]) for row in manifest}
    phenotype_set = set(phenotypes)
    positive_keys: set[tuple[str, str]] = set()
    for row in _read_predictions(predictions_path):
        note_id = str(row["note_id"])
        phenotype = str(row["phenotype"])
        if phenotype not in phenotype_set:
            continue
        if evaluation_scope != "all" and split_by_note.get(note_id) != evaluation_scope:
            continue
        try:
            prediction = _binary(row["prediction"])
        except ValueError:
            continue
        if prediction == 1:
            positive_keys.add((note_id, phenotype))

    audit_keys = _read_audit_keys(evidence_audit_path)
    audited_positive_keys = positive_keys & audit_keys
    unaudited_positive_keys = positive_keys - audit_keys
    coverage = len(audited_positive_keys) / len(positive_keys) if positive_keys else None
    coverage_pass = (
        coverage is not None
        and coverage >= audit_policy.min_positive_prediction_coverage
        and not unaudited_positive_keys
        if audit_policy.min_positive_prediction_coverage == 1.0
        else coverage is not None and coverage >= audit_policy.min_positive_prediction_coverage
    )

    human = dict(summary.get("human_evidence_audit", {}))
    human.update(
        {
            "positive_predictions_requiring_audit": len(positive_keys),
            "positive_predictions_audited": len(audited_positive_keys),
            "positive_prediction_audit_coverage": coverage,
            "required_positive_prediction_audit_coverage": audit_policy.min_positive_prediction_coverage,
            "unaudited_positive_predictions": len(unaudited_positive_keys),
        }
    )
    summary["human_evidence_audit"] = human

    acceptance = dict(summary.get("acceptance", {}))
    human_checks = dict(acceptance.get("human_gate_checks", {}))
    human_checks["positive_prediction_audit_coverage"] = bool(coverage_pass)
    acceptance["human_gate_checks"] = human_checks

    automated_pass = bool(acceptance.get("automated_gate_pass"))
    audit_available = evidence_audit_path is not None and bool(audit_keys)
    quality_checks_pass = bool(human_checks.get("evidence_support")) and bool(
        human_checks.get("severe_context_error")
    )
    strict_human_pass = audit_available and coverage_pass and quality_checks_pass
    acceptance["human_gate_pass"] = strict_human_pass

    if not automated_pass:
        final_status = "FAIL_AUTOMATED_GATE"
    elif not audit_available:
        final_status = "PASS_AUTOMATED_PENDING_HUMAN_EVIDENCE_AUDIT"
    elif strict_human_pass:
        final_status = "PASS"
    else:
        final_status = "FAIL_HUMAN_EVIDENCE_GATE"
    acceptance["final_status"] = final_status
    acceptance["strict_positive_audit_coverage_enforced"] = True
    summary["acceptance"] = acceptance
    return summary, metrics, errors, manifest
