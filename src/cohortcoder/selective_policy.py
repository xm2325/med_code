from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta


@dataclass(frozen=True)
class SelectivePolicySelection:
    threshold: float | None
    n_auto: int
    coverage: float
    empirical_accuracy: float | None
    one_sided_lower_bound: float | None
    target_accuracy: float
    alpha: float
    min_auto: int

    def to_dict(self) -> dict:
        return asdict(self)


def one_sided_binomial_lower_bound(successes: int, n: int, *, alpha: float = 0.05) -> float | None:
    """Exact one-sided lower confidence bound for a binomial success probability.

    This is used only as a conservative policy-selection criterion on held-out policy
    calibration data. It does not turn retrieval confidence into a calibrated probability.
    """
    successes = int(successes)
    n = int(n)
    if n <= 0:
        return None
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, n - successes + 1))


def select_threshold_by_accuracy_lower_bound(
    predictions: pd.DataFrame,
    *,
    target_accuracy: float = 0.95,
    alpha: float = 0.05,
    min_auto: int = 20,
) -> SelectivePolicySelection:
    """Maximise policy-calibration coverage subject to a conservative accuracy bound.

    Threshold selection uses only the supplied policy-calibration frame. The caller is
    responsible for keeping this frame disjoint from model-selection and TEST data.
    """
    if not 0 <= target_accuracy <= 1:
        raise ValueError("target_accuracy must be between 0 and 1")
    if min_auto < 1:
        raise ValueError("min_auto must be >= 1")
    if predictions.empty or not {"confidence", "correct"}.issubset(predictions.columns):
        return SelectivePolicySelection(None, 0, 0.0, None, None, target_accuracy, alpha, min_auto)

    working = predictions.copy()
    working["_confidence"] = pd.to_numeric(working["confidence"], errors="coerce")
    working["_correct"] = pd.to_numeric(working["correct"], errors="coerce").fillna(0).astype(int)
    working = working.dropna(subset=["_confidence"]).reset_index(drop=True)
    if working.empty:
        return SelectivePolicySelection(None, 0, 0.0, None, None, target_accuracy, alpha, min_auto)

    feasible: list[SelectivePolicySelection] = []
    for threshold in sorted(working["_confidence"].unique()):
        accepted = working[working["_confidence"] >= threshold]
        n_auto = int(len(accepted))
        if n_auto < min_auto:
            continue
        successes = int(accepted["_correct"].sum())
        empirical = float(successes / n_auto)
        lower = one_sided_binomial_lower_bound(successes, n_auto, alpha=alpha)
        if lower is not None and lower + 1e-12 >= target_accuracy:
            feasible.append(
                SelectivePolicySelection(
                    threshold=float(threshold),
                    n_auto=n_auto,
                    coverage=float(n_auto / len(working)),
                    empirical_accuracy=empirical,
                    one_sided_lower_bound=lower,
                    target_accuracy=float(target_accuracy),
                    alpha=float(alpha),
                    min_auto=int(min_auto),
                )
            )

    if not feasible:
        return SelectivePolicySelection(None, 0, 0.0, None, None, target_accuracy, alpha, min_auto)

    feasible.sort(
        key=lambda item: (
            item.n_auto,
            item.one_sided_lower_bound or -np.inf,
            item.empirical_accuracy or -np.inf,
            -(item.threshold or 0.0),
        ),
        reverse=True,
    )
    return feasible[0]


def apply_frozen_threshold(predictions: pd.DataFrame, threshold: float | None) -> dict:
    """Evaluate one already-selected threshold without changing it on TEST."""
    if predictions.empty:
        return {"threshold": threshold, "n_auto": 0, "coverage": 0.0, "accuracy_at_1": None}
    if threshold is None:
        return {"threshold": None, "n_auto": 0, "coverage": 0.0, "accuracy_at_1": None}
    confidence = pd.to_numeric(predictions["confidence"], errors="coerce")
    accepted = predictions[confidence >= float(threshold)]
    return {
        "threshold": float(threshold),
        "n_auto": int(len(accepted)),
        "coverage": float(len(accepted) / len(predictions)),
        "accuracy_at_1": (
            float(pd.to_numeric(accepted["correct"], errors="coerce").fillna(0).mean())
            if len(accepted)
            else None
        ),
    }
