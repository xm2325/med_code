from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterable, Mapping

from .llm import ExternalLLMPolicyError


def validate_multi_candidate_payload(payload: Mapping[str, Any], *, allowed_codes: Iterable[str], allowed_evidence_quotes: Iterable[str]) -> tuple[bool, list[str]]:
    allowed = [str(code) for code in allowed_codes]
    allowed_quotes = {str(q) for q in allowed_evidence_quotes}
    errors: list[str] = []
    ranked = payload.get("ranked_codes")
    if not isinstance(ranked, list):
        errors.append("ranked_codes_not_list")
        ranked = []
    ranked = [str(x) for x in ranked]
    if len(ranked) != len(set(ranked)):
        errors.append("duplicate_ranked_codes")
    if set(ranked) != set(allowed):
        errors.append("candidate_set_changed")
    rationales = payload.get("candidate_rationales")
    if not isinstance(rationales, list):
        errors.append("candidate_rationales_not_list")
        rationales = []
    rationale_codes = [str(item.get("code", "")) for item in rationales if isinstance(item, Mapping)]
    if len(rationale_codes) != len(set(rationale_codes)):
        errors.append("duplicate_rationale_codes")
    if set(rationale_codes) != set(allowed):
        errors.append("rationale_candidate_set_changed")
    for item in rationales:
        if not isinstance(item, Mapping):
            errors.append("invalid_rationale_item")
            continue
        if not str(item.get("rationale", "")).strip():
            errors.append("missing_candidate_rationale")
        quotes = item.get("evidence_quotes", [])
        if not isinstance(quotes, list):
            errors.append("evidence_quotes_not_list")
            continue
        if not quotes:
            errors.append("candidate_rationale_missing_real_evidence")
        if any(str(q) not in allowed_quotes for q in quotes):
            errors.append("non_verbatim_or_unapproved_evidence")
    return not errors, sorted(set(errors))


class DeepSeekRealCandidateEvaluator:
    """One-call-per-case fixed-candidate reranking plus grounded rationale for every option."""

    def __init__(self, *, api_key: str | None = None, model: str = "deepseek-v4-pro", base_url: str = "https://api.deepseek.com/chat/completions", timeout_seconds: int = 90) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        self.model = str(model)
        self.base_url = str(base_url)
        self.timeout_seconds = int(timeout_seconds)

    def _request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages, "response_format": {"type": "json_object"}, "temperature": 0, "max_tokens": 2200, "stream": False}
        req = urllib.request.Request(self.base_url, data=json.dumps(body).encode("utf-8"), headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            outer = json.loads(response.read().decode("utf-8"))
        return json.loads(outer["choices"][0]["message"]["content"])

    def evaluate_case(self, *, phrase: str, candidates: list[dict[str, Any]], candidate_grounding: list[dict[str, Any]], allow_external_llm: bool, data_classification: str, max_attempts: int = 2) -> dict[str, Any]:
        if not allow_external_llm:
            raise ExternalLLMPolicyError("Real-data DeepSeek evaluation requires explicit consent")
        if str(data_classification).lower() != "public":
            raise ExternalLLMPolicyError("This evaluator is restricted to explicitly public data")
        allowed_codes = [str(item["code"]) for item in candidates]
        allowed_quotes = [str(phrase)]
        grounding_by_code = {str(item.get("code")): item for item in candidate_grounding}
        candidate_payload = []
        for candidate in candidates:
            code = str(candidate["code"])
            grounding = grounding_by_code.get(code, {})
            candidate_payload.append({"code": code, "term": str(candidate.get("term", "")), "retrieval_score": float(candidate.get("score", 0.0) or 0.0), "allowed_source_evidence": [str(phrase)], "terminology_support": grounding.get("terminology_support", {}), "historical_support": grounding.get("historical_support", [])[:2]})
        system = (
            "You rerank a FIXED set of medical coding candidates for a PUBLIC benchmark phrase. Preserve exactly the supplied code set. "
            "Provide a separate concise rationale for EVERY candidate, including weaker alternatives. Every rationale MUST quote at least one exact string from allowed_source_evidence. "
            "A weaker candidate rationale should explain mismatch or ambiguity rather than pretending the code is correct. Do not invent clinical facts. Return JSON only."
        )
        user = {"benchmark_phrase": str(phrase), "candidates": candidate_payload, "required_schema": {"ranked_codes": allowed_codes, "overall_uncertainty": "brief ambiguity statement", "candidate_rationales": [{"code": code, "rationale": "support or mismatch reasoning grounded in phrase", "evidence_quotes": [str(phrase)]} for code in allowed_codes]}}
        last_errors: list[str] = []
        for _ in range(max(1, int(max_attempts))):
            try:
                payload = self._request([{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}])
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_errors = [f"api_or_parse_error:{type(exc).__name__}"]
                continue
            valid, errors = validate_multi_candidate_payload(payload, allowed_codes=allowed_codes, allowed_evidence_quotes=allowed_quotes)
            if valid:
                return {"accepted": True, "payload": payload, "validation_errors": [], "model": self.model}
            last_errors = errors
        return {"accepted": False, "payload": None, "validation_errors": last_errors or ["unknown_failure"], "model": self.model}
