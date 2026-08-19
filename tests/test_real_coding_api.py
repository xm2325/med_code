from typing import Any, Mapping, Sequence

import pytest

from cohortcoder.real_coding import RealCodingEngine, validate_frozen_rerank


class FakeJSONClient:
    model, model_revision = "fake-qwen", "test-revision"

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def complete_json(self, messages: Sequence[Mapping[str, str]], *, schema=None, max_tokens=800):
        self.calls.append({"messages": list(messages), "schema": schema})
        return self.responses.pop(0)


TEXT = "After methotrexate the patient developed nausea."
EVENT = {"quote": "nausea", "start": 41, "end": 47, "assertion": "affirmed"}


def test_prompt_allowlist_and_audit():
    fake = FakeJSONClient([{"events": [{**EVENT, "code": "10028813"}]}])
    result = RealCodingEngine(fake).prompt_only(TEXT)
    assert result["events"][0]["selected"]["llt_term"] == "Nausea"
    assert result["status"] == "LIVE"
    assert result["audit"]["model_revision"] == "test-revision"
    assert result["causality_not_assessed"] is True
    assert result["events"][0]["selected"]["retrieval_score"] == 1.0
    assert result["events"][0]["offset_source"] == "SERVER_DERIVED_FROM_QUOTE_LENGTH"
    prompt_schema = fake.calls[0]["schema"]["properties"]["events"]["items"]
    assert "end" not in prompt_schema["properties"]
    assert "Do not return end" in fake.calls[0]["messages"][0]["content"]


def test_hallucinated_code_is_not_repaired():
    raw = {"events": [{**EVENT, "code": "99999999"}]}
    result = RealCodingEngine(FakeJSONClient([raw])).prompt_only(TEXT)
    assert result["events"] == [] and result["raw_model_json"] == raw
    assert "event_0:code_not_in_allowlist" in result["validation_errors"]


def test_exact_offsets_required_and_lexical_retrieval():
    bad = RealCodingEngine(FakeJSONClient([{"events": [{**EVENT, "start": 0, "end": 6}]}])).extract_lexical(TEXT)
    assert "event_0:non_verbatim_span" in bad["validation_errors"]
    good = RealCodingEngine(FakeJSONClient([{"events": [EVENT]}])).extract_lexical(TEXT)
    assert good["events"][0]["selected"]["llt_code"] == "10028813"
    assert good["events"][0]["offset_source"] == "SERVER_DERIVED_FROM_QUOTE_LENGTH"


def test_extractor_is_never_allowed_to_supply_a_code():
    injected = {"events": [{**EVENT, "code": "99999999"}]}
    result = RealCodingEngine(FakeJSONClient([injected])).extract_lexical(TEXT)
    assert result["events"] == []
    assert "event_0:extractor_must_not_supply_code" in result["validation_errors"]


def test_hybrid_frozen_set_and_rejection():
    okay = [{"events": [EVENT]}, {"ranked_codes": ["10028813"], "selected_code": "10028813"}]
    result = RealCodingEngine(FakeJSONClient(okay)).hybrid_rag(TEXT)
    assert result["events"][0]["selected"]["llt_code"] == "10028813"
    bad = [{"events": [EVENT]}, {"ranked_codes": ["99999999"], "selected_code": "99999999"}]
    rejected = RealCodingEngine(FakeJSONClient(bad)).hybrid_rag(TEXT)
    assert rejected["events"][0]["selected"] is None
    assert "rerank:candidate_set_changed" in rejected["validation_errors"]


def test_strict_selected_code_and_not_run():
    valid, errors = validate_frozen_rerank({"ranked_codes": ["A", "B"], "selected_code": "C"}, ["A", "B"])
    assert not valid and "selected_code_outside_candidate_set" in errors
    fake = FakeJSONClient([{"events": [{**EVENT, "code": "10028813"}]}])
    result = RealCodingEngine(fake).run(TEXT, methods=["prompt_only"])
    assert result["methods"]["extract_lexical"] == {"status": "NOT_RUN"}
    assert "code" not in result["methods"]["extract_lexical"]


def test_unknown_method():
    with pytest.raises(ValueError, match="unknown methods"):
        RealCodingEngine(FakeJSONClient([])).run(TEXT, methods=["made_up"])


def test_method_failure_does_not_abort_other_methods():
    fake = FakeJSONClient([RuntimeError("boom"), {"events": [EVENT]}])
    original = fake.complete_json
    def raising(*args, **kwargs):
        value = fake.responses.pop(0)
        if isinstance(value, Exception): raise value
        return value
    fake.complete_json = raising
    result = RealCodingEngine(fake).run(TEXT, methods=["prompt_only", "extract_lexical"])
    assert result["methods"]["prompt_only"]["status"] == "ERROR"
    assert result["methods"]["extract_lexical"]["status"] == "LIVE"
    assert result["execution_status"] == "PARTIAL"


def test_hybrid_retrieval_is_independent_and_structured():
    engine = RealCodingEngine(FakeJSONClient([]))
    candidates = engine._hybrid_retrieve("nausea", 3)
    assert candidates[0]["llt_code"] == "10028813"
    assert candidates[0]["pt_code"] is None
    assert candidates[0]["retrieval_score"] > 0
    assert candidates[0]["score_semantics"] == "retrieval_relevance_not_probability"


def test_fastapi_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from cohortcoder.api import create_app
    fake = FakeJSONClient([{"events": [{**EVENT, "code": "10028813"}]}])
    client = TestClient(create_app(RealCodingEngine(fake)))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/v1/meta").json()["meddra_version"] == "28.0"
    assert len(client.get("/records").json()) == 50
    response = client.post("/api/v1/coding/run", json={"schema_version":"1.0","meddra_version":"28.0",
        "data_classification":"synthetic","text": TEXT, "methods": ["prompt_only"]})
    assert response.status_code == 200
