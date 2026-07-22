from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _parse_spans(value: object) -> list[tuple[int, int]]:
    if isinstance(value, list):
        raw = value
    else:
        try:
            raw = json.loads(str(value or "[]"))
        except (TypeError, json.JSONDecodeError):
            return []
    spans: list[tuple[int, int]] = []
    if not isinstance(raw, list):
        return spans
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            spans.append((int(item[0]), int(item[1])))
        except (TypeError, ValueError):
            continue
    return spans


def apply_task_input_spans(
    explanations: list[dict[str, Any]],
    source_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Prefer exact task-provided mention spans for display evidence when available.

    This function does not use the gold code. CADEC mention spans are part of the task
    input and are treated as provenance for *where the input mention came from*, not as a
    gold rationale proving that the predicted code is correct.
    """
    if len(explanations) != len(source_rows):
        raise ValueError("explanations and source_rows must be positionally aligned")
    out: list[dict[str, Any]] = []
    source = source_rows.reset_index(drop=True)
    for index, original in enumerate(explanations):
        item = dict(original)
        row = source.iloc[index]
        text = str(item.get("text", "") or row.get("text", ""))
        spans = _parse_spans(row.get("spans_json", ""))
        valid = [(a, b) for a, b in spans if 0 <= a < b <= len(text)]
        if valid:
            evidence = []
            for start, end in valid:
                evidence.append({
                    "start": start,
                    "end": end,
                    "quote": text[start:end],
                    "source": "task_input_span",
                    "score": 3.0,
                    "matched_phrases": [],
                })
            item["evidence_spans"] = evidence
            item["evidence_quotes"] = [span["quote"] for span in evidence]
            item["evidence_verbatim"] = True
            item["task_input_span_used"] = True
        else:
            item["task_input_span_used"] = False
        out.append(item)
    return out
