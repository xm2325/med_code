from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterable, Mapping


class ExternalLLMPolicyError(RuntimeError):
    pass


def validate_llm_rationale(
    payload: Mapping[str, Any],
    *,
    locked_code: str,
    allowed_quotes: Iterable[str],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if str(payload.get("code", "")) != str(locked_code):
        errors.append("code_changed")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("missing_rationale")
    quotes = payload.get("evidence_quotes")
    if not isinstance(quotes, list):
        errors.append("evidence_quotes_not_list")
        quotes = []
    allowed = {str(quote) for quote in allowed_quotes}
    if not allowed:
        errors.append("no_affirmed_evidence_available")
    for quote in quotes:
        if str(quote) not in allowed:
            errors.append("non_verbatim_or_unapproved_evidence")
            break
    return not errors, errors


class DeepSeekRationaleClient:
    """Generate a rationale without allowing the LLM to alter the selected code.

    By default only already-grounded evidence spans and terminology knowledge are sent
    to the external API, not the full clinical note. Restricted/private clinical data
    are blocked from external transmission in this client.
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
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = int(timeout_seconds)
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")

    @staticmethod
    def _check_data_policy(*, allow_external_llm: bool, data_classification: str) -> None:
        if not allow_external_llm:
            raise ExternalLLMPolicyError("External LLM use requires explicit --allow-external-llm consent")
        classification = str(data_classification).strip().lower()
        if classification not in {"public", "synthetic"}:
            raise ExternalLLMPolicyError(
                "External DeepSeek calls are blocked for restricted/private clinical data. "
                "Use a locally governed model or an approved institutional endpoint instead."
            )

    def _request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 800,
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
        if not content:
            raise RuntimeError("DeepSeek returned empty content")
        return json.loads(content)

    def generate(
        self,
        explanation: Mapping[str, Any],
        *,
        allow_external_llm: bool,
        data_classification: str,
        few_shot_examples: list[Mapping[str, Any]] | None = None,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        self._check_data_policy(
            allow_external_llm=allow_external_llm,
            data_classification=data_classification,
        )
        code = str(explanation.get("predicted_code", ""))
        term = str(explanation.get("predicted_term", ""))
        system = str(explanation.get("coding_system", ""))
        evidence_quotes = [str(value) for value in explanation.get("evidence_quotes", [])]
        if not evidence_quotes:
            return {
                "accepted": False,
                "payload": None,
                "validation_errors": ["no_affirmed_evidence_available"],
                "model": self.model,
            }
        knowledge = explanation.get("external_knowledge", {})

        examples = []
        for item in (few_shot_examples or [])[:3]:
            examples.append({
                "code": str(item.get("code", "")),
                "term": str(item.get("term", "")),
                "evidence_quotes": item.get("evidence_quotes", item.get("evidence_quote", [])),
                "rationale": str(item.get("rationale", "")),
            })

        system_prompt = (
            "You write concise clinical coding rationales in JSON. The code is LOCKED and must not be changed. "
            "Use only the supplied verbatim evidence quotes and terminology knowledge. Do not infer a new diagnosis, "
            "severity, causal relationship, temporality, or treatment not explicitly supported. Return JSON with keys: "
            "code, rationale, evidence_quotes, uncertainty, knowledge_support. evidence_quotes must be copied exactly "
            "from the allowed list."
        )
        user_payload = {
            "task": "Explain why the locked coding label is supported by the supplied evidence.",
            "locked_code": code,
            "coding_system": system,
            "term": term,
            "allowed_evidence_quotes": evidence_quotes,
            "terminology_knowledge": knowledge,
            "few_shot_training_examples": examples,
            "required_json_example": {
                "code": code,
                "rationale": "Concise explanation grounded only in the evidence.",
                "evidence_quotes": evidence_quotes[:1],
                "uncertainty": "State any ambiguity or missing evidence.",
                "knowledge_support": "How terminology knowledge supports the mapping without adding new facts.",
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
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
                last_errors = [f"api_or_parse_error:{type(exc).__name__}"]
                continue
            valid, errors = validate_llm_rationale(payload, locked_code=code, allowed_quotes=evidence_quotes)
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
            "validation_errors": last_errors or ["unknown_generation_failure"],
            "model": self.model,
        }


def apply_deepseek_rationales(
    explanations: list[dict[str, Any]],
    client: DeepSeekRationaleClient,
    *,
    allow_external_llm: bool,
    data_classification: str,
    few_shot_examples: list[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    enhanced: list[dict[str, Any]] = []
    for original in explanations:
        item = dict(original)
        original_status = str(item.get("explanation_status", ""))
        original_source = str(item.get("explanation_source", "grounded_deterministic"))
        result = client.generate(
            item,
            allow_external_llm=allow_external_llm,
            data_classification=data_classification,
            few_shot_examples=few_shot_examples,
        )
        item["llm_validation"] = result
        if result["accepted"]:
            payload = result["payload"]
            item["why"] = str(payload["rationale"])
            item["llm_uncertainty"] = str(payload.get("uncertainty", ""))
            item["llm_knowledge_support"] = str(payload.get("knowledge_support", ""))
            item["llm_evidence_quotes"] = list(payload.get("evidence_quotes", []))
            item["explanation_source"] = f"deepseek:{result['model']}"
            item["llm_rationale_status"] = "validated"
            item["explanation_status"] = original_status
        else:
            item["explanation_source"] = original_source
            item["llm_rationale_status"] = "rejected_fallback_to_deterministic"
            item["explanation_status"] = original_status
        enhanced.append(item)
    return enhanced
