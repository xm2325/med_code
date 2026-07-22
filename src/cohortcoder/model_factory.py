from __future__ import annotations

from typing import Any

import pandas as pd

from .advanced import (
    AdvancedModelConfig,
    AdvancedSingleLabelCoder,
    CrossEncoderCandidateReranker,
    DenseSemanticIndex,
)
from .core import HistoricalCoder


def build_singlelabel_coder_from_policy(
    history: pd.DataFrame,
    terminology: pd.DataFrame,
    frozen_policy: dict[str, Any],
    *,
    device: str | None = None,
):
    """Rebuild the exact single-label coding family recorded in frozen_policy.json."""
    model_type = str(frozen_policy.get("model_type", "lexical_historical"))
    history_weight = float(frozen_policy.get("history_weight", 0.5))
    if model_type != "advanced_singlelabel":
        return HistoricalCoder(history_weight=history_weight, top_k=10).fit(history, terminology)

    dense_weight = float(frozen_policy.get("dense_weight", 0.0) or 0.0)
    reranker_weight = float(frozen_policy.get("reranker_weight", 0.0) or 0.0)
    dense_model_name = frozen_policy.get("dense_model_name")
    cross_encoder_model_name = frozen_policy.get("cross_encoder_model_name")

    if dense_weight > 0 and not dense_model_name:
        raise ValueError("Frozen advanced policy requires dense_model_name when dense_weight > 0")
    if reranker_weight > 0 and not cross_encoder_model_name:
        raise ValueError("Frozen advanced policy requires cross_encoder_model_name when reranker_weight > 0")

    dense = DenseSemanticIndex(str(dense_model_name), device=device) if dense_weight > 0 else None
    reranker = (
        CrossEncoderCandidateReranker(str(cross_encoder_model_name), device=device)
        if reranker_weight > 0
        else None
    )
    config = AdvancedModelConfig(
        history_weight=history_weight,
        dense_weight=dense_weight,
        reranker_weight=reranker_weight,
    )
    return AdvancedSingleLabelCoder(config, dense_index=dense, reranker=reranker).fit(history, terminology)
