from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def evaluate_explanation_quality(explanation: dict[str, Any]) -> dict[str, Any]:
    text = str(explanation.get("text", "") or "")
    quotes = [str(value) for value in explanation.get("evidence_quotes", [])]
    verbatim = bool(quotes) and all(quote in text for quote in quotes)
    context_review_required = bool(explanation.get("context_review_required", False))
    status = str(explanation.get("explanation_status", ""))
    knowledge = explanation.get("external_knowledge", {}) or {}
    knowledge_present = bool(str(knowledge.get("term", "") or "").strip())
    faithfulness = explanation.get("faithfulness", {}) or {}
    original = faithfulness.get("original_code_score")
    removed = faithfulness.get("evidence_removed_code_score")
    comprehensiveness_positive = None
    if original is not None and removed is not None:
        comprehensiveness_positive = float(original) - float(removed) > 0

    failures: list[str] = []
    warnings: list[str] = []
    if not quotes:
        failures.append("no_evidence_quotes")
    if quotes and not verbatim:
        failures.append("non_verbatim_evidence")
    if context_review_required:
        failures.append("clinical_context_requires_review")
    if status in {"insufficient_grounding", "insufficient_affirmed_evidence"}:
        failures.append(status)
    if not knowledge_present:
        warnings.append("missing_terminology_knowledge")
    if comprehensiveness_positive is False:
        warnings.append("evidence_removal_did_not_reduce_score")
    if comprehensiveness_positive is None:
        warnings.append("faithfulness_not_available")

    gate = "FAIL" if failures else ("WARN" if warnings else "PASS")
    return {
        "gate": gate,
        "failures": failures,
        "warnings": warnings,
        "verbatim_evidence": verbatim,
        "knowledge_present": knowledge_present,
        "comprehensiveness_positive": comprehensiveness_positive,
    }


def apply_explanation_quality_gate(explanations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a conservative quality gate and only allow decision downgrades.

    The gate can change AUTO_CANDIDATE/CODE_PROPOSAL to HUMAN_REVIEW but can never
    promote a record from review to automatic handling.
    """
    out: list[dict[str, Any]] = []
    for original in explanations:
        item = deepcopy(original)
        quality = evaluate_explanation_quality(item)
        item["explanation_quality"] = quality
        previous = str(item.get("decision", ""))
        item["decision_before_explanation_quality_gate"] = previous
        if quality["gate"] == "FAIL" and previous in {"AUTO_CANDIDATE", "CODE_PROPOSAL"}:
            item["decision"] = "HUMAN_REVIEW"
            item["explanation_quality_decision_override"] = True
        else:
            item["explanation_quality_decision_override"] = False
        out.append(item)
    return out


def summarize_explanation_quality(explanations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(explanations)
    gates = [str((item.get("explanation_quality") or {}).get("gate", "")) for item in items]
    n = len(items)
    auto_like = {"AUTO_CANDIDATE", "CODE_PROPOSAL"}
    before = [str(item.get("decision_before_explanation_quality_gate", item.get("decision", ""))) for item in items]
    after = [str(item.get("decision", "")) for item in items]
    n_before_auto = sum(value in auto_like for value in before)
    n_after_auto = sum(value in auto_like for value in after)
    return {
        "n": n,
        "pass_rate": gates.count("PASS") / n if n else 0.0,
        "warn_rate": gates.count("WARN") / n if n else 0.0,
        "fail_rate": gates.count("FAIL") / n if n else 0.0,
        "n_decisions_downgraded": sum(bool(item.get("explanation_quality_decision_override")) for item in items),
        "automatic_or_proposal_rate_before_quality_gate": n_before_auto / n if n else 0.0,
        "automatic_or_proposal_rate_after_quality_gate": n_after_auto / n if n else 0.0,
        "additional_review_rate_due_to_explanation_gate": (n_before_auto - n_after_auto) / n if n else 0.0,
    }


def write_explanation_quality_artifacts(output_dir: str | Path, explanations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(explanations)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        quality = item.get("explanation_quality") or evaluate_explanation_quality(item)
        rows.append({
            "record_id": str(item.get("record_id", "")),
            "predicted_code": str(item.get("predicted_code", "")),
            "predicted_term": str(item.get("predicted_term", "")),
            "quality_gate": str(quality.get("gate", "")),
            "quality_failures_json": json.dumps(quality.get("failures", [])),
            "quality_warnings_json": json.dumps(quality.get("warnings", [])),
            "verbatim_evidence": bool(quality.get("verbatim_evidence", False)),
            "knowledge_present": bool(quality.get("knowledge_present", False)),
            "comprehensiveness_positive": quality.get("comprehensiveness_positive"),
            "decision_before_gate": str(item.get("decision_before_explanation_quality_gate", "")),
            "decision_after_gate": str(item.get("decision", "")),
            "decision_downgraded": bool(item.get("explanation_quality_decision_override", False)),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "explanation_quality_cases.csv", index=False)
    summary = summarize_explanation_quality(items)
    (output / "explanation_quality.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
