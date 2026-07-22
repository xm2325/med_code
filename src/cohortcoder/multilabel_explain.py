from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .explain import build_explanation_record
from .multilabel import MultiLabelHistoricalCoder, _candidate_list, parse_code_list


def historical_support_for_codes(
    coder: MultiLabelHistoricalCoder,
    text: str,
    codes: list[str],
    *,
    limit: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve raw TRAIN examples only for explanation provenance.

    Ranking itself uses sparse code centroids. Raw note retrieval is delayed until a
    small set of codes has already been proposed for explanation.
    """
    query_text = str(text or "").strip() or "__empty__"
    scores = cosine_similarity(
        coder.history_vectorizer.transform([query_text]),
        coder.history_matrix,
    )[0]
    order = np.argsort(-scores)
    wanted = {str(code) for code in codes}
    result: dict[str, list[dict[str, Any]]] = {code: [] for code in wanted}
    for idx in order:
        document_codes = set(coder.history_codes[int(idx)])
        for code in wanted & document_codes:
            if len(result[code]) >= int(limit):
                continue
            result[code].append({
                "text": str(coder.history.iloc[int(idx)]["text"]),
                "code": code,
                "term": str(coder.code_to_term.get(code, "")),
                "similarity": float(scores[int(idx)]),
            })
        if all(len(result[code]) >= int(limit) for code in wanted):
            break
    return result


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
    Faithfulness scores are computed by the same centroid-based model used for ranking.
    """
    if not {"record_id", "text"}.issubset(records.columns):
        raise ValueError("records require record_id and text")
    by_record = {str(row.record_id): row for _, row in records.iterrows()}
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
        selected_codes = [str(item.get("code", "")) for item in selected if str(item.get("code", ""))]
        support_by_code = historical_support_for_codes(coder, str(record.text), selected_codes) if selected_codes else {}
        for candidate in selected:
            code = str(candidate.get("code", ""))
            if not code:
                continue
            mapping = {
                "record_id": record_id,
                "text": str(record.text),
                "mention": "",
                "predicted_code": code,
                "predicted_term": str(candidate.get("term", "")),
                "confidence": float(candidate.get("score", 0.0) or 0.0),
                "decision": "CODE_PROPOSAL",
                "coding_system": "ICD-10",
                "historical_cases_json": json.dumps(support_by_code.get(code, [])),
            }
            item = build_explanation_record(mapping, terminology, coder=coder)
            item["gold_code_present"] = code in set(parse_code_list(pred_row.get("gold_codes_json", "[]")))
            item["proposal_score"] = float(candidate.get("score", 0.0) or 0.0)
            explanations.append(item)
    return explanations
