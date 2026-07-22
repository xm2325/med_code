from cohortcoder.explanation_quality import apply_explanation_quality_gate, evaluate_explanation_quality


def test_quality_gate_downgrades_auto_when_evidence_missing():
    item = {
        "text": "No rash was reported.",
        "evidence_quotes": [],
        "explanation_status": "insufficient_affirmed_evidence",
        "context_review_required": True,
        "external_knowledge": {"term": "Rash"},
        "faithfulness": {},
        "decision": "AUTO_CANDIDATE",
    }
    out = apply_explanation_quality_gate([item])[0]
    assert out["explanation_quality"]["gate"] == "FAIL"
    assert out["decision"] == "HUMAN_REVIEW"
    assert out["explanation_quality_decision_override"] is True


def test_quality_gate_never_promotes_human_review():
    item = {
        "text": "severe muscle pain",
        "evidence_quotes": ["severe muscle pain"],
        "explanation_status": "grounded",
        "context_review_required": False,
        "external_knowledge": {"term": "Myalgia"},
        "faithfulness": {
            "original_code_score": 0.8,
            "evidence_removed_code_score": 0.2,
        },
        "decision": "HUMAN_REVIEW",
    }
    quality = evaluate_explanation_quality(item)
    assert quality["gate"] == "PASS"
    out = apply_explanation_quality_gate([item])[0]
    assert out["decision"] == "HUMAN_REVIEW"
    assert out["explanation_quality_decision_override"] is False
