from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UncertaintyReference:
    """Validation-derived reference thresholds for conservative uncertainty flags.

    These thresholds are diagnostics, not proof that a record is truly out-of-distribution.
    They must be fitted without TEST labels and are intended to make routing more conservative.
    """

    confidence_floor: float
    margin_floor: float
    top_score_floor: float
    fitted_on: str = "validation_correct_predictions"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_candidates(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    try:
        payload = json.loads(str(value or "[]"))
    except (json.JSONDecodeError, TypeError):
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _candidate_summary(value: object) -> tuple[float, float]:
    candidates = _parse_candidates(value)
    scores = []
    for item in candidates:
        try:
            scores.append(float(item.get("score", 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)
    scores.sort(reverse=True)
    top = scores[0] if scores else 0.0
    second = scores[1] if len(scores) > 1 else 0.0
    return float(top), float(max(0.0, top - second))


def fit_uncertainty_reference(
    validation_predictions: pd.DataFrame,
    *,
    lower_quantile: float = 0.05,
) -> UncertaintyReference:
    """Fit diagnostic floors using only correctly coded validation examples.

    The method deliberately uses a simple empirical reference rather than claiming a
    clinically validated OOD detector. TEST labels must never be used here.
    """

    if not 0.0 <= float(lower_quantile) < 0.5:
        raise ValueError("lower_quantile must be in [0, 0.5)")
    frame = validation_predictions.copy()
    if "correct" in frame.columns:
        correct = pd.to_numeric(frame["correct"], errors="coerce").fillna(0).astype(int) == 1
        reference = frame[correct].copy()
        if reference.empty:
            reference = frame.copy()
    else:
        reference = frame.copy()
    if reference.empty:
        return UncertaintyReference(0.0, 0.0, 0.0)

    confidence = pd.to_numeric(reference.get("confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    summaries = [_candidate_summary(value) for value in reference.get("candidates_json", pd.Series("[]", index=reference.index))]
    top_scores = np.asarray([item[0] for item in summaries], dtype=float)
    margins = np.asarray([item[1] for item in summaries], dtype=float)
    q = float(lower_quantile)
    return UncertaintyReference(
        confidence_floor=float(np.quantile(confidence, q)),
        margin_floor=float(np.quantile(margins, q)),
        top_score_floor=float(np.quantile(top_scores, q)),
    )


def annotate_uncertainty(
    predictions: pd.DataFrame,
    reference: UncertaintyReference,
    *,
    train_codes: Iterable[str] = (),
) -> pd.DataFrame:
    """Attach auditable uncertainty flags without using held-out gold correctness.

    `possible_distribution_shift` is intentionally a warning label, not a formal OOD
    diagnosis. It becomes true when multiple weak-support signals are present.
    """

    frame = predictions.copy()
    known_codes = {str(code) for code in train_codes if str(code)}
    top_scores: list[float] = []
    margins: list[float] = []
    for value in frame.get("candidates_json", pd.Series("[]", index=frame.index)):
        top, margin = _candidate_summary(value)
        top_scores.append(top)
        margins.append(margin)
    frame["uncertainty_top_score"] = top_scores
    frame["uncertainty_margin"] = margins
    confidence = pd.to_numeric(frame.get("confidence", 0.0), errors="coerce").fillna(0.0)
    frame["flag_low_confidence"] = confidence < reference.confidence_floor
    frame["flag_low_margin"] = frame["uncertainty_margin"] < reference.margin_floor
    frame["flag_low_candidate_support"] = frame["uncertainty_top_score"] < reference.top_score_floor
    if known_codes and "predicted_code" in frame.columns:
        frame["flag_code_unseen_in_train"] = ~frame["predicted_code"].astype(str).isin(known_codes)
    else:
        frame["flag_code_unseen_in_train"] = False
    signal_columns = [
        "flag_low_confidence",
        "flag_low_margin",
        "flag_low_candidate_support",
        "flag_code_unseen_in_train",
    ]
    frame["uncertainty_flag_count"] = frame[signal_columns].astype(int).sum(axis=1)
    frame["possible_distribution_shift"] = frame["uncertainty_flag_count"] >= 2
    frame["uncertainty_interpretation"] = np.where(
        frame["possible_distribution_shift"],
        "Multiple weak-support signals; route conservatively and review for possible distribution shift.",
        "No multi-signal distribution-shift warning under the frozen validation reference.",
    )
    return frame


def apply_uncertainty_routing(frame: pd.DataFrame) -> pd.DataFrame:
    """Only downgrade automatic decisions when the uncertainty audit warns."""

    out = frame.copy()
    if "decision" not in out.columns:
        out["decision"] = "HUMAN_REVIEW"
    out["decision_before_uncertainty_gate"] = out["decision"].astype(str)
    warning = out.get("possible_distribution_shift", pd.Series(False, index=out.index)).astype(bool)
    can_auto = out["decision"].astype(str).isin(["AUTO_CANDIDATE", "CODE_PROPOSAL"])
    override = warning & can_auto
    out.loc[override, "decision"] = "HUMAN_REVIEW"
    out["uncertainty_decision_override"] = override
    return out


def uncertainty_summary(frame: pd.DataFrame) -> dict[str, Any]:
    n = int(len(frame))
    warnings = frame.get("possible_distribution_shift", pd.Series(False, index=frame.index)).astype(bool)
    overrides = frame.get("uncertainty_decision_override", pd.Series(False, index=frame.index)).astype(bool)
    return {
        "n": n,
        "possible_distribution_shift_rate": float(warnings.mean()) if n else 0.0,
        "n_possible_distribution_shift": int(warnings.sum()),
        "n_decisions_downgraded": int(overrides.sum()),
        "interpretation": "Diagnostic warning only; not a validated clinical OOD detector.",
    }
