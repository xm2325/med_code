from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd


def _candidate_codes(value: object) -> list[str]:
    """Return candidate codes from the JSON payload written by the benchmark."""
    if isinstance(value, list):
        items = value
    else:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        try:
            items = json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            return []
    codes: list[str] = []
    for item in items:
        if isinstance(item, dict) and "code" in item:
            codes.append(str(item["code"]))
        elif isinstance(item, str):
            codes.append(item)
    return codes


def choose_threshold_max_coverage(predictions: pd.DataFrame, target_accuracy: float) -> float | None:
    """Choose the validation threshold with maximum coverage subject to target accuracy.

    This is intentionally different from taking the first high-confidence threshold
    that reaches the target. The study question is how much manual review can be
    reduced while preserving a prespecified agreement target, so coverage must be
    maximised on validation data only.
    """
    if not 0 <= target_accuracy <= 1:
        raise ValueError("target_accuracy must be between 0 and 1")
    if predictions.empty or not {"confidence", "correct"}.issubset(predictions.columns):
        return None

    working = predictions.copy()
    working["_confidence"] = pd.to_numeric(working["confidence"], errors="coerce")
    working["_correct"] = pd.to_numeric(working["correct"], errors="coerce").fillna(0)
    working = working.dropna(subset=["_confidence"])
    feasible: list[tuple[int, float, float]] = []
    for threshold in sorted(working["_confidence"].unique()):
        accepted = working[working["_confidence"] >= threshold]
        if accepted.empty:
            continue
        accuracy = float(accepted["_correct"].mean())
        if accuracy + 1e-12 >= target_accuracy:
            feasible.append((len(accepted), accuracy, float(threshold)))
    if not feasible:
        return None

    # Primary objective: maximum validation coverage. Secondary objective: higher
    # validation accuracy. Final tie-break: lower threshold for deterministic output.
    feasible.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    return feasible[0][2]


def annotate_prediction_diagnostics(
    predictions: pd.DataFrame,
    seen_codes: Iterable[str],
    candidate_k: int = 10,
) -> pd.DataFrame:
    """Add seen/unseen status, gold rank, and failure taxonomy to predictions."""
    out = predictions.copy()
    seen = {str(code) for code in seen_codes}
    ranks: list[float] = []
    novelty: list[str] = []
    error_types: list[str] = []

    for _, row in out.iterrows():
        gold = str(row.get("gold_code", ""))
        predicted = str(row.get("predicted_code", ""))
        codes = _candidate_codes(row.get("candidates_json", "[]"))[:candidate_k]
        rank = codes.index(gold) + 1 if gold in codes else None
        ranks.append(float(rank) if rank is not None else np.nan)
        novelty.append("seen_code" if gold in seen else "unseen_code")
        if predicted == gold:
            error_types.append("correct")
        elif rank is None:
            error_types.append("candidate_generation_failure")
        else:
            error_types.append("ranking_failure")

    out["gold_candidate_rank"] = ranks
    out["code_novelty"] = novelty
    out["error_type"] = error_types
    return out


def subgroup_metrics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarise exact accuracy and candidate recall overall and by code novelty."""
    groups = [
        ("all", diagnostics),
        ("seen_code", diagnostics[diagnostics["code_novelty"] == "seen_code"]),
        ("unseen_code", diagnostics[diagnostics["code_novelty"] == "unseen_code"]),
    ]
    rows: list[dict] = []
    for name, group in groups:
        if group.empty:
            rows.append({
                "subgroup": name,
                "n": 0,
                "accuracy_at_1": None,
                "candidate_recall_at_5": None,
                "candidate_recall_at_10": None,
            })
            continue
        ranks = pd.to_numeric(group["gold_candidate_rank"], errors="coerce")
        rows.append({
            "subgroup": name,
            "n": int(len(group)),
            "accuracy_at_1": float(pd.to_numeric(group["correct"], errors="coerce").fillna(0).mean()),
            "candidate_recall_at_5": float((ranks <= 5).fillna(False).mean()),
            "candidate_recall_at_10": float((ranks <= 10).fillna(False).mean()),
        })
    return pd.DataFrame(rows)


def coverage_accuracy_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    """Descriptive held-out coverage/accuracy curve over all observed thresholds."""
    if predictions.empty:
        return pd.DataFrame(columns=["threshold", "coverage", "accuracy", "n_auto"])
    working = predictions.copy()
    working["confidence"] = pd.to_numeric(working["confidence"], errors="coerce")
    working["correct"] = pd.to_numeric(working["correct"], errors="coerce").fillna(0)
    working = working.dropna(subset=["confidence"])
    rows = []
    for threshold in sorted(working["confidence"].unique(), reverse=True):
        accepted = working[working["confidence"] >= threshold]
        rows.append({
            "threshold": float(threshold),
            "coverage": float(len(accepted) / len(working)),
            "accuracy": float(accepted["correct"].mean()),
            "n_auto": int(len(accepted)),
        })
    return pd.DataFrame(rows)


def policy_stress_test(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    targets: Iterable[float] = (0.90, 0.95, 0.98, 0.99),
) -> pd.DataFrame:
    """Select each threshold on validation, then evaluate the frozen policy on TEST."""
    rows = []
    for target in targets:
        threshold = choose_threshold_max_coverage(validation_predictions, float(target))
        if threshold is None:
            rows.append({
                "target_accuracy": float(target),
                "validation_threshold": None,
                "test_coverage": 0.0,
                "test_accuracy": None,
                "test_n_auto": 0,
                "target_met_on_test": False,
            })
            continue
        accepted = test_predictions[pd.to_numeric(test_predictions["confidence"], errors="coerce") >= threshold]
        accuracy = float(pd.to_numeric(accepted["correct"], errors="coerce").fillna(0).mean()) if len(accepted) else None
        rows.append({
            "target_accuracy": float(target),
            "validation_threshold": float(threshold),
            "test_coverage": float(len(accepted) / len(test_predictions)) if len(test_predictions) else 0.0,
            "test_accuracy": accuracy,
            "test_n_auto": int(len(accepted)),
            "target_met_on_test": bool(accuracy is not None and accuracy + 1e-12 >= float(target)),
        })
    return pd.DataFrame(rows)


def failure_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Count correct, candidate-generation, and ranking outcomes by seen/unseen status."""
    rows = []
    for novelty in ["all", "seen_code", "unseen_code"]:
        subset = diagnostics if novelty == "all" else diagnostics[diagnostics["code_novelty"] == novelty]
        counts = subset["error_type"].value_counts().to_dict()
        for error_type in ["correct", "ranking_failure", "candidate_generation_failure"]:
            count = int(counts.get(error_type, 0))
            rows.append({
                "subgroup": novelty,
                "error_type": error_type,
                "n": count,
                "rate": float(count / len(subset)) if len(subset) else None,
            })
    return pd.DataFrame(rows)
