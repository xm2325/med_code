from cohortcoder.deepseek_real_eval import validate_multi_candidate_payload


def test_every_candidate_requires_grounded_rationale_and_fixed_code_set():
    payload = {
        "ranked_codes": ["B", "A"],
        "overall_uncertainty": "A and B are close",
        "candidate_rationales": [
            {"code": "A", "rationale": "A is supported by the phrase.", "evidence_quotes": ["muscle pain"]},
            {"code": "B", "rationale": "B is less specific for the phrase.", "evidence_quotes": ["muscle pain"]},
        ],
    }
    valid, errors = validate_multi_candidate_payload(payload, allowed_codes=["A", "B"], allowed_evidence_quotes=["muscle pain"])
    assert valid is True and errors == []

    bad = dict(payload)
    bad["candidate_rationales"] = [{"code":"A","rationale":"x","evidence_quotes":[]}]
    valid2, errors2 = validate_multi_candidate_payload(bad, allowed_codes=["A", "B"], allowed_evidence_quotes=["muscle pain"])
    assert valid2 is False
    assert "candidate_rationale_missing_real_evidence" in errors2 or "rationale_candidate_set_changed" in errors2
