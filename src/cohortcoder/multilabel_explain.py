from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .explain import build_explanation_record
from .multilabel import MultiLabelHistoricalCoder, _candidate_list, parse_code_list


class MultiLabelCodeScoreAdapter:
    """Expose per-code scores to the shared faithfulness evaluator."""

    def __init__(self, coder: MultiLabelHistoricalCoder):
        self.coder = coder
        self.code_index = {
            str(code): idx for idx, code in enumerate(coder.terminology["code"].astype(str))
        }

    def score_code(self, text: str, code: str) -> float:
        code = str(code)
        idx = self.code_index.get(code)
        if idx is None:
            return 0.0
        query_text = str(text or "").strip() or "__empty__"
        term_score = float(cosine_similarity(
            self.coder.term_vectorizer.transform([query_text]),
            self.coder.term_matrix[idx],
        )[0, 0])
        hist_scores = cosine_similarity(
            self.coder.history_vectorizer.transform([query_text]),
            self.coder.history_matrix,
        )[0]
        history_score = 0.0
        for hist_idx, similarity in enumerate(hist_scores):
            if code in self.coder.history_codes[hist_idx]:
                history_score = max(history_score, float(similarity))
        return (1.0 - self.coder.history_weight) * term_score + self.coder.history_weight * history_score


def historical_support_for_code(
    coder: MultiLabelHistoricalCoder,
    text: str,
    code: str,
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    query_text = str(text or "").strip() or "__empty__"
    scores = cosine_similarity(
        coder.history_vectorizer.transform([query_text]),
        coder.history_matrix,
    )[0]
    order = np.argsort(-scores)
    rows: list[dict[str, Any]] = []
    for idx in order:
        if str(code) not in coder.history_codes[int(idx)]:
            continue
        rows.append({
            "text": str(coder.history.iloc[int(idx)]["text"]),
            "code": str(code),
            "term": str(coder.code_to_term.get(str(code), "")),
            "similarity": float(scores[int(idx)]),
        })
        if len(rows) >= int(limit):
            break
    return rows


def explain_multilabel_proposals(
    records: pd.DataFrame,
    predictions: pd.DataFrame,
    terminology: pd.DataFrame,
    coder: MultiLabelHistoricalCoder,
    *,
    threshold: float | None,
    fallback_top_k: int = 5,
) -> list[dict[str, Any]]:
    """Build one grounded explanation per proposed ICD code.

    A threshold-selected code proposal is not equivalent to automatic coding of the
    entire note. Explanations therefore use the decision label ``CODE_PROPOSAL``.
    """
    if not {"record_id", "text"}.issubset(records.columns):
        raise ValueError("records require record_id and text")
    by_record = {str(row.record_id): row for _, row in records.iterrows()}
    scorer = MultiLabelCodeScoreAdapter(coder)
    explanations: list[dict[str, Any]] = []

    for _, pred_row in predictions.iterrows():
        record_id = str(pred_row["record_id"])
        if record_id not in by_record:
            raise ValueError(f"Prediction record not found in source records: {record_id}")
        record = by_record[record_id]
        candidates = _candidate_list(pred_row["candidates_json"])
        if threshold is None:
            selected = candidates[: int(fallback_top_k)]
        else:
            selected = [
                item for item in candidates
                if float(item.get("score", 0.0) or 0.0) >= float(threshold)
            ]
        for candidate in selected:
            code = str(candidate.get("code", ""))
            if not code:
                continue
            support = historical_support_for_code(coder, str(record.text), code)
            mapping = {
                "record_id": record_id,
                "text": str(record.text),
                "mention": "",
                "predicted_code": code,
                "predicted_term": str(candidate.get("term", "")),
                "confidence": float(candidate.get("score", 0.0) or 0.0),
                "decision": "CODE_PROPOSAL",
                "coding_system": "ICD-10",
                "historical_cases_json": json.dumps(support),
            }
            item = build_explanation_record(mapping, terminology, coder=scorer)
            item["gold_code_present"] = code in set(parse_code_list(pred_row.get("gold_codes_json", "[]")))
            item["proposal_score"] = float(candidate.get("score", 0.0) or 0.0)
            explanations.append(item)
    return explanations
