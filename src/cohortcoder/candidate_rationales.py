from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

import pandas as pd

from .explain import extract_evidence_spans


def _terminology_row(terminology: pd.DataFrame, code: str) -> Mapping[str, Any]:
    matched = terminology[terminology["code"].astype(str) == str(code)]
    return {} if matched.empty else matched.iloc[0].to_dict()


def _history_for_code(history_items: Iterable[Mapping[str, Any]], code: str, limit: int = 2) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in history_items:
        if str(item.get("code", "")) != str(code):
            continue
        out.append({
            "text": str(item.get("text", "")),
            "code": str(code),
            "term": str(item.get("term", "")),
            "similarity": float(item.get("similarity", 0.0) or 0.0),
        })
        if len(out) >= limit:
            break
    return out


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return [dict(item) for item in parsed if isinstance(item, Mapping)] if isinstance(parsed, list) else []


def build_candidate_rationales(
    *,
    text: str,
    mention: str,
    candidates: list[dict[str, Any]],
    terminology: pd.DataFrame,
    historical_cases: list[dict[str, Any]] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Build one grounded evidence/rationale object for every displayed candidate.

    The rationale is deterministic and explicitly distinguishes source-text evidence,
    terminology support, and historical expert-coded provenance. It never invents evidence.
    """
    history = historical_cases or []
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates[: max(1, int(top_k))], start=1):
        code = str(candidate.get("code", ""))
        term_row = _terminology_row(terminology, code)
        term = str(term_row.get("term", candidate.get("term", "")) or "")
        synonyms = str(term_row.get("synonyms", "") or "")
        definition = str(term_row.get("definition", "") or "")
        hierarchy = str(term_row.get("hierarchy", "") or "")
        source = str(term_row.get("knowledge_source", "") or "")
        spans = extract_evidence_spans(
            str(text or ""),
            mention=str(mention or ""),
            term=term,
            synonyms=synonyms,
            max_spans=3,
        )
        evidence = [span.to_dict() for span in spans]
        quotes = [span.quote for span in spans]
        historical = _history_for_code(history, code)
        support_parts = []
        if quotes:
            support_parts.append("Source evidence: " + "; ".join(f'“{q}”' for q in quotes[:2]))
        else:
            support_parts.append("No exact supporting source span was grounded for this candidate.")
        if term:
            support_parts.append(f"Terminology mapping: {code} — {term}.")
        if definition:
            support_parts.append("Terminology definition is available as supporting knowledge.")
        if historical:
            support_parts.append(f"{len(historical)} similar TRAIN historical expert-coded example(s) support this code as provenance.")
        rows.append({
            "rank": rank,
            "code": code,
            "term": term,
            "model_score": float(candidate.get("score", 0.0) or 0.0),
            "evidence_spans": evidence,
            "evidence_quotes": quotes,
            "rationale": " ".join(support_parts),
            "terminology_support": {
                "term": term,
                "synonyms": synonyms,
                "definition": definition,
                "hierarchy": hierarchy,
                "knowledge_source": source,
            },
            "historical_support": historical,
            "grounded": bool(quotes),
        })
    return rows


def build_review_packet(
    record: Mapping[str, Any],
    terminology: pd.DataFrame,
    *,
    route: str,
    uncertainty: Mapping[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    candidates = _parse_json_list(record.get("candidates_json", []))
    history = _parse_json_list(record.get("historical_cases_json", []))
    rationales = build_candidate_rationales(
        text=str(record.get("text", "") or ""),
        mention=str(record.get("mention", "") or ""),
        candidates=candidates,
        terminology=terminology,
        historical_cases=history,
        top_k=top_k,
    )
    return {
        "record_id": str(record.get("record_id", "")),
        "text": str(record.get("text", "") or ""),
        "mention": str(record.get("mention", "") or ""),
        "predicted_code": str(record.get("predicted_code", "")),
        "predicted_term": str(record.get("predicted_term", "")),
        "confidence": float(record.get("confidence", 0.0) or 0.0),
        "route": str(route),
        "uncertainty": dict(uncertainty),
        "candidate_options": rationales,
        "human_selected_code": "",
        "human_selection_reason": "",
    }
