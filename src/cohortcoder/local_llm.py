from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


class JSONChatClient(Protocol):
    model: str
    model_revision: str

    def complete_json(self, messages: Sequence[Mapping[str, str]], *, schema: Mapping[str, Any] | None = None,
                      max_tokens: int = 800) -> dict[str, Any]: ...


@dataclass
class OpenAICompatibleQwenClient:
    """JSON-only client for a locally governed vLLM OpenAI endpoint."""

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen3.8-27b-meddra"
    model_revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    api_key: str = "local"
    timeout_seconds: int = 120

    def complete_json(self, messages: Sequence[Mapping[str, str]], *, schema: Mapping[str, Any] | None = None,
                      max_tokens: int = 800) -> dict[str, Any]:
        response_format: dict[str, Any] = {"type": "json_object"}
        if schema:
            response_format = {"type": "json_schema", "json_schema": {
                "name": "medcode_result", "strict": True, "schema": dict(schema)}}
        body = {"model": self.model, "messages": [dict(m) for m in messages], "temperature": 0,
                "top_p": 1, "seed": 0, "max_tokens": int(max_tokens), "stream": False,
                "response_format": response_format, "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            envelope = json.loads(response.read().decode("utf-8"))
        content = envelope["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("local model returned empty content")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("local model JSON response must be an object")
        return payload
