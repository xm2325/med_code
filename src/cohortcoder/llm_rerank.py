from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterable

from .llm import ExternalLLMPolicyError


def validate_rerank_payload(payload: dict[str, Any], allowed_codes: Iterable[str]) -> tuple[bool, list[str]]:
    allowed = [str(code) for code in allowed_codes]
    ranked = payload.get("ranked_codes")
    errors: list[str] = []
    if not isinstance(ranked, list):
        return False, ["ranked_codes_not_list"]
    ranked = [str(code) for code in ranked]
    if len(ranked) != len(set(ranked)):
        errors.append("duplicate_codes")
    if set(ranked) != set(allowed):
        errors.append("candidate_set_changed")
    return not errors, errors


class DeepSeekCandidateReranker:
    """Rerank an already-frozen candidate set without inventing new codes.

    This client is intentionally separate from model selection. A prompt/model policy
    should be fixed on development/validation data before the held-out TEST candidate
    set is submitted for final evaluation.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "deepseek-v4-pro",
        base_url: str = "https://api.deepseek.com/chat/completions",
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = str(model)
        self.base_url = str(base_url)
        self.timeout_seconds = int(timeout_seconds)
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")

    @staticmethod
    def _check_policy(*, allow_external_llm: bool, data_classification: str) -> None:
        if not allow_external_llm:
            raise ExternalLLMPolicyError("External LLM reranking requires explicit consent")
        if str(data_classification).strip().lower() not in {"public", "synthetic"}:
            raise ExternalLLMPolicyError(
                "External DeepSeek reranking is blocked for restricted/private clinical data"
            )

    def _request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 1000,
            "stream": False,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        return json.loads(content)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        allow_external_llm: bool,
        data_classification: str,
        evidence_quotes: list[str] | None = None,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        self._check_policy(
            allow_external_llm=allow_external_llm,
            data_classification=data_classification,
        )
        allowed_codes = [str(item["code"]) for item in candidates]
        candidate_payload = [
            {
                "code": str(item["code"]),
                "term": str(item.get("term", "")),
                "retrieval_score": item.get("score", item.get("retrieval_score")),
            }
            for item in candidates
        ]
        system_prompt = (
            "You rerank a fixed list of medical coding candidates. You MUST preserve exactly the supplied code set, "
            "must not invent a new code, and must not infer facts absent from the supplied mention/evidence. Return JSON "
            "with ranked_codes (all supplied codes exactly once), top_choice_reason, and uncertainty."
        )
        user_payload = {
            "query_or_mention": str(query),
            "approved_evidence_quotes": [str(value) for value in (evidence_quotes or [])],
            "candidates": candidate_payload,
            "required_output": {
                "ranked_codes": allowed_codes,
                "top_choice_reason": "Brief evidence-grounded reason.",
                "uncertainty": "Any ambiguity between candidates.",
            },
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        last_errors: list[str] = []
        for _ in range(max(1, int(max_attempts))):
            try:
                payload = self._request(messages)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_errors = [f"api_or_parse_error:{type(exc).__name__}"]
                continue
            valid, errors = validate_rerank_payload(payload, allowed_codes)
            if valid:
                return {
                    "accepted": True,
                    "payload": payload,
                    "validation_errors": [],
                    "model": self.model,
                }
            last_errors = errors
        return {
            "accepted": False,
            "payload": None,
            "validation_errors": last_errors or ["unknown_rerank_failure"],
            "model": self.model,
        }
