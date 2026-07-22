from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Iterable

import numpy as np


def _normalised_probabilities(candidates: Iterable[dict[str, Any]], temperature: float = 1.0) -> np.ndarray:
    scores = np.asarray([float(item.get("score", 0.0) or 0.0) for item in candidates], dtype=float)
    if scores.size == 0:
        return np.asarray([], dtype=float)
    temperature = max(float(temperature), 1e-6)
    shifted = (scores - scores.max()) / temperature
    exp = np.exp(np.clip(shifted, -50, 50))
    total = float(exp.sum())
    return exp / total if total > 0 else np.ones_like(exp) / len(exp)


def candidate_uncertainty(candidates: list[dict[str, Any]], *, temperature: float = 1.0) -> dict[str, float | None]:
    """Return auditable uncertainty features without pretending they are calibrated probabilities."""
    if not candidates:
        return {
            "top_score": None,
            "score_margin": None,
            "normalised_entropy": None,
            "top_softmax_share": None,
        }
    scores = [float(item.get("score", 0.0) or 0.0) for item in candidates]
    probs = _normalised_probabilities(candidates, temperature=temperature)
    entropy = -float(sum(float(p) * log(float(p) + 1e-12) for p in probs))
    max_entropy = log(len(probs)) if len(probs) > 1 else 1.0
    return {
        "top_score": scores[0],
        "score_margin": scores[0] - scores[1] if len(scores) > 1 else scores[0],
        "normalised_entropy": entropy / max_entropy if len(probs) > 1 else 0.0,
        "top_softmax_share": float(probs[0]),
    }


@dataclass(frozen=True)
class ReviewRoutingPolicy:
    auto_threshold: float = 0.85
    topk_choice_threshold: float = 0.45
    max_entropy_for_auto: float = 0.55
    min_margin_for_auto: float = 0.08
    top_k: int = 5

    def route(
        self,
        *,
        confidence: float,
        uncertainty: dict[str, float | None],
        explanation_gate: str | None = None,
        ood_flag: bool = False,
    ) -> str:
        """Route to AUTO, TOP_K_HUMAN_CHOICE, or FULL_EXPERT_REVIEW.

        Explanation/OOD checks may only make routing more conservative.
        """
        if ood_flag or str(explanation_gate or "").upper() == "FAIL":
            return "FULL_EXPERT_REVIEW"
        entropy = uncertainty.get("normalised_entropy")
        margin = uncertainty.get("score_margin")
        if (
            float(confidence) >= self.auto_threshold
            and (entropy is None or float(entropy) <= self.max_entropy_for_auto)
            and (margin is None or float(margin) >= self.min_margin_for_auto)
        ):
            return "AUTO_CANDIDATE"
        if float(confidence) >= self.topk_choice_threshold:
            return "TOP_K_HUMAN_CHOICE"
        return "FULL_EXPERT_REVIEW"


def simple_ood_flag(
    *,
    top_score: float | None,
    terminology_overlap: float | None = None,
    history_overlap: float | None = None,
    min_top_score: float = 0.05,
) -> bool:
    """Conservative heuristic OOD flag for development diagnostics.

    This is not a validated medical OOD detector. It is deliberately explicit so a stronger
    detector can replace it without changing the routing contract.
    """
    if top_score is None or float(top_score) < min_top_score:
        return True
    if terminology_overlap is not None and history_overlap is not None:
        return float(terminology_overlap) <= 0 and float(history_overlap) <= 0
    return False
