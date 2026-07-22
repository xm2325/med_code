from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd


def _parse_spans(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    merged: list[list[int]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _length(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _overlap_length(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> int:
    a = _merge_intervals(left)
    b = _merge_intervals(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def validate_rationale_offsets(annotations: pd.DataFrame, records: pd.DataFrame) -> dict[str, Any]:
    required = {"record_id", "code", "start", "end"}
    if not required.issubset(annotations.columns):
        raise ValueError(f"rationale annotations require {sorted(required)}")
    if not {"record_id", "text"}.issubset(records.columns):
        raise ValueError("records require record_id and text")
    text_by_record = records.drop_duplicates("record_id").set_index("record_id")["text"].astype(str).to_dict()
    missing_record = invalid_bounds = quote_mismatch = 0
    for _, row in annotations.iterrows():
        record_id = str(row["record_id"])
        text = text_by_record.get(record_id)
        if text is None:
            missing_record += 1
            continue
        start, end = int(row["start"]), int(row["end"])
        if not (0 <= start < end <= len(text)):
            invalid_bounds += 1
            continue
        if "quote" in annotations.columns and str(row.get("quote", "")) and text[start:end] != str(row["quote"]):
            quote_mismatch += 1
    return {
        "annotations": int(len(annotations)),
        "missing_record": int(missing_record),
        "invalid_bounds": int(invalid_bounds),
        "quote_mismatch": int(quote_mismatch),
        "valid": not any([missing_record, invalid_bounds, quote_mismatch]),
    }


def evaluate_rationale_overlap(predicted: pd.DataFrame, reference: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Character-overlap evaluation for evidence spans by record and code."""
    required = {"record_id", "code", "start", "end"}
    if not required.issubset(reference.columns):
        raise ValueError(f"reference annotations require {sorted(required)}")
    if not {"record_id", "predicted_code"}.issubset(predicted.columns):
        raise ValueError("predicted explanations require record_id and predicted_code")

    reference_groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for _, row in reference.iterrows():
        key = (str(row["record_id"]), str(row["code"]))
        reference_groups.setdefault(key, []).append((int(row["start"]), int(row["end"])))

    predicted_groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for _, row in predicted.iterrows():
        key = (str(row["record_id"]), str(row["predicted_code"]))
        spans = _parse_spans(row.get("evidence_spans", row.get("evidence_spans_json", "[]")))
        predicted_groups[key] = [
            (int(span.get("start", 0)), int(span.get("end", 0)))
            for span in spans
            if int(span.get("end", 0)) > int(span.get("start", 0))
        ]

    rows: list[dict[str, Any]] = []
    for key, gold_spans in reference_groups.items():
        pred_spans = predicted_groups.get(key, [])
        gold_len = _length(gold_spans)
        pred_len = _length(pred_spans)
        overlap = _overlap_length(gold_spans, pred_spans)
        precision = overlap / pred_len if pred_len else 0.0
        recall = overlap / gold_len if gold_len else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "record_id": key[0],
            "code": key[1],
            "char_precision": float(precision),
            "char_recall": float(recall),
            "char_f1": float(f1),
            "any_overlap": bool(overlap > 0),
            "prediction_present": key in predicted_groups,
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {
            "n_reference_record_code_pairs": 0,
            "prediction_pair_coverage": 0.0,
            "macro_char_precision": None,
            "macro_char_recall": None,
            "macro_char_f1": None,
            "any_overlap_rate": None,
        }, detail
    return {
        "n_reference_record_code_pairs": int(len(detail)),
        "prediction_pair_coverage": float(detail["prediction_present"].mean()),
        "macro_char_precision": float(detail["char_precision"].mean()),
        "macro_char_recall": float(detail["char_recall"].mean()),
        "macro_char_f1": float(detail["char_f1"].mean()),
        "any_overlap_rate": float(detail["any_overlap"].mean()),
    }, detail
