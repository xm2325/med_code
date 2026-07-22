from cohortcoder.llm import DeepSeekRationaleClient


def test_generate_rejects_before_api_when_no_affirmed_evidence(monkeypatch):
    client = DeepSeekRationaleClient(api_key="synthetic-key")

    def fail_if_called(_messages):
        raise AssertionError("API should not be called without affirmed evidence")

    monkeypatch.setattr(client, "_request", fail_if_called)
    result = client.generate(
        {
            "predicted_code": "A",
            "predicted_term": "Example",
            "coding_system": "DEMO",
            "evidence_quotes": [],
            "external_knowledge": {},
        },
        allow_external_llm=True,
        data_classification="synthetic",
    )
    assert result["accepted"] is False
    assert result["validation_errors"] == ["no_affirmed_evidence_available"]
