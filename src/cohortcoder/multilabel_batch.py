from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .multilabel import MultiLabelHistoricalCoder


def rank_dataframe_batched(
    coder: MultiLabelHistoricalCoder,
    records: pd.DataFrame,
    *,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Rank ICD candidates in memory-bounded batches.

    Dense score matrices exist only for ``batch_size × number_of_codes`` rather than for
    the entire TEST set at once. Only top-K candidates are serialized per record.
    """
    if not {"record_id", "text", "gold_codes_json"}.issubset(records.columns):
        raise ValueError("records require record_id, text, gold_codes_json")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    code_values = coder.terminology["code"].astype(str).tolist()
    term_values = coder.terminology["term"].astype(str).tolist()
    top_k = min(coder.top_k, len(code_values))
    rows = []
    reset = records.reset_index(drop=True)

    for start in range(0, len(reset), int(batch_size)):
        batch = reset.iloc[start:start + int(batch_size)]
        texts = [str(value or "").strip() or "__empty__" for value in batch["text"]]
        term_queries = coder.term_vectorizer.transform(texts)
        history_queries = coder.history_vectorizer.transform(texts)
        term_scores = cosine_similarity(term_queries, coder.term_matrix)
        history_scores = cosine_similarity(history_queries, coder.code_history_matrix)
        combined = (1.0 - coder.history_weight) * term_scores + coder.history_weight * history_scores

        if top_k == len(code_values):
            candidate_indices = np.argsort(-combined, axis=1)
        else:
            partition = np.argpartition(-combined, kth=top_k - 1, axis=1)[:, :top_k]
            candidate_indices = np.take_along_axis(
                partition,
                np.argsort(-np.take_along_axis(combined, partition, axis=1), axis=1),
                axis=1,
            )

        for local_idx, (_, record) in enumerate(batch.iterrows()):
            candidates = []
            for idx in candidate_indices[local_idx, :top_k]:
                code_idx = int(idx)
                candidates.append({
                    "code": code_values[code_idx],
                    "term": term_values[code_idx],
                    "score": float(combined[local_idx, code_idx]),
                    "terminology_score": float(term_scores[local_idx, code_idx]),
                    "history_score": float(history_scores[local_idx, code_idx]),
                })
            rows.append({
                "record_id": str(record["record_id"]),
                "subject_id": str(record.get("subject_id", "")),
                "gold_codes_json": str(record["gold_codes_json"]),
                "candidates_json": json.dumps(candidates),
            })
    return pd.DataFrame(rows)
