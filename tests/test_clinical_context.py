from cohortcoder.clinical_context import audit_explanation_context, classify_assertion


def test_negated_rash_is_not_affirmative_support():
    assert classify_assertion("No rash was reported at follow-up.") == "negated"


def test_uncertain_diagnosis_is_not_affirmative_support():
    assert classify_assertion("Possible pneumonia; chest imaging pending.") == "uncertain"


def test_context_guard_forces_review_when_only_evidence_is_negated():
    explanation = {
        "record_id": "r1",
        "coding_system": "DEMO",
        "predicted_code": "R1",
        "predicted_term": "Rash",
        "decision": "AUTO_CANDIDATE",
        "why": "Initial explanation",
        "text": "At follow-up, no rash was reported.",
        "evidence_quotes": ["rash"],
        "evidence_spans": [{"start": 17, "end": 21, "quote": "rash", "source": "supporting_sentence", "score": 1.0}],
    }
    audited = audit_explanation_context([explanation])[0]
    assert audited["evidence_quotes"] == []
    assert audited["context_review_required"] is True
    assert audited["decision"] == "HUMAN_REVIEW"
    assert audited["explanation_status"] == "insufficient_affirmed_evidence"
