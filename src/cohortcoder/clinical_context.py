from __future__ import annotations

import re
from typing import Any


NEGATION = re.compile(r"\b(no|not|denies|denied|without|negative for|never|no evidence of|free of)\b", re.I)
UNCERTAINTY = re.compile(r"\b(possible|possibly|probable|suspected|suspect|query|question of|may have|might have|rule out|r/o)\b", re.I)
FAMILY = re.compile(r"\b(family history|mother|father|sister|brother|maternal|paternal)\b", re.I)
RESOLVED = re.compile(r"\b(history of|previous|previously|past history|resolved|remote history)\b", re.I)


def classify_assertion(context: str) -> str:
    """Conservative lightweight assertion classification for rationale safety.

    This is not a replacement for a validated clinical assertion model. It is a guard
    that prevents obvious negated/uncertain/family-history text from being presented as
    affirmative support without review.
    """
    text = str(context or "")
    if FAMILY.search(text):
        return "family_history"
    if NEGATION.search(text):
        return "negated"
    if UNCERTAINTY.search(text):
        return "uncertain"
    if RESOLVED.search(text):
        return "historical_or_resolved"
    return "affirmed"


def _local_context(text: str, start: int, end: int, window: int = 100) -> str:
    left_boundary = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
    left = max(left_boundary + 1, start - window, 0)
    dot = text.find(".", end)
    newline = text.find("\n", end)
    candidates = [value for value in [dot, newline] if value >= 0]
    right_boundary = min(candidates) + 1 if candidates else min(len(text), end + window)
    right = min(max(right_boundary, end), len(text))
    return text[left:right].strip()


def audit_explanation_context(explanations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark assertion context and keep only affirmed spans as supporting evidence."""
    audited: list[dict[str, Any]] = []
    for original in explanations:
        item = dict(original)
        text = str(item.get("text", ""))
        spans = []
        affirmed_quotes: list[str] = []
        excluded = []
        for raw in item.get("evidence_spans", []):
            span = dict(raw)
            start = int(span.get("start", 0))
            end = int(span.get("end", start))
            context = _local_context(text, start, end)
            assertion = classify_assertion(context)
            span["clinical_assertion"] = assertion
            span["assertion_context"] = context
            spans.append(span)
            if assertion == "affirmed":
                affirmed_quotes.append(str(span.get("quote", "")))
            else:
                excluded.append({
                    "quote": str(span.get("quote", "")),
                    "clinical_assertion": assertion,
                    "context": context,
                })

        item["evidence_spans"] = spans
        item["all_extracted_evidence_quotes"] = [str(span.get("quote", "")) for span in spans]
        item["evidence_quotes"] = [quote for quote in affirmed_quotes if quote]
        item["excluded_evidence_context"] = excluded
        item["context_review_required"] = not bool(item["evidence_quotes"])

        if item["context_review_required"]:
            item["explanation_status"] = "insufficient_affirmed_evidence"
            item["decision"] = "HUMAN_REVIEW"
            item["why"] = (
                f"The model proposed {item.get('coding_system', 'the coding system')} code "
                f"{item.get('predicted_code', '')} ({item.get('predicted_term', '')}), but the extracted "
                "text is negated, uncertain, historical/family-context, or otherwise lacks a clearly affirmed "
                "supporting span. The code should not be auto-accepted from this explanation."
            )
        elif excluded:
            item["why"] = str(item.get("why", "")) + (
                " Some nearby extracted text was excluded from supporting evidence because its clinical "
                "assertion context was not clearly affirmative."
            )
        audited.append(item)
    return audited
