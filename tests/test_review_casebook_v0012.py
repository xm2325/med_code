import pandas as pd

from cohortcoder.review_casebook import build_review_casebook


def test_casebook_prioritizes_auto_and_high_confidence_errors():
    predictions = pd.DataFrame([
        {"record_id": "r1", "gold_code": "A", "predicted_code": "B", "confidence": 0.99, "correct": 0, "decision": "AUTO_CANDIDATE", "code_novelty": "seen_code"},
        {"record_id": "r2", "gold_code": "A", "predicted_code": "A", "confidence": 0.95, "correct": 1, "decision": "AUTO_CANDIDATE", "code_novelty": "seen_code"},
        {"record_id": "r3", "gold_code": "C", "predicted_code": "D", "confidence": 0.80, "correct": 0, "decision": "HUMAN_REVIEW", "code_novelty": "unseen_code"},
    ])
    casebook = build_review_casebook(predictions, max_cases=3)
    reasons = set(casebook["review_reason"])
    assert "auto_candidate_error" in reasons
    assert any(reason in reasons for reason in ["high_confidence_error", "unseen_code_error"])
    assert "expert_code_correct" in casebook.columns
