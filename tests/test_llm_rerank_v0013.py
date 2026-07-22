from cohortcoder.llm_rerank import validate_rerank_payload


def test_rerank_payload_must_preserve_candidate_set():
    valid, errors = validate_rerank_payload({"ranked_codes": ["B", "A"]}, ["A", "B"])
    assert valid is True
    assert errors == []

    valid, errors = validate_rerank_payload({"ranked_codes": ["A", "C"]}, ["A", "B"])
    assert valid is False
    assert "candidate_set_changed" in errors


def test_rerank_payload_rejects_duplicates():
    valid, errors = validate_rerank_payload({"ranked_codes": ["A", "A"]}, ["A", "B"])
    assert valid is False
    assert "duplicate_codes" in errors
