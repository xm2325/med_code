import pandas as pd

from cohortcoder.candidate_rationales import build_candidate_rationales
from cohortcoder.uncertainty import ReviewRoutingPolicy, candidate_uncertainty


def test_uncertainty_margin_and_routing():
    candidates = [
        {"code": "A", "term": "Alpha", "score": 0.9},
        {"code": "B", "term": "Beta", "score": 0.3},
    ]
    u = candidate_uncertainty(candidates)
    assert round(u["score_margin"], 6) == 0.6
    policy = ReviewRoutingPolicy(auto_threshold=0.8, topk_choice_threshold=0.4, min_margin_for_auto=0.1)
    assert policy.route(confidence=0.9, uncertainty=u) == "AUTO_CANDIDATE"
    assert policy.route(confidence=0.6, uncertainty=u) == "TOP_K_HUMAN_CHOICE"
    assert policy.route(confidence=0.2, uncertainty=u) == "FULL_EXPERT_REVIEW"
    assert policy.route(confidence=0.99, uncertainty=u, explanation_gate="FAIL") == "FULL_EXPERT_REVIEW"


def test_every_displayed_candidate_has_separate_grounded_rationale_object():
    terminology = pd.DataFrame([
        {"code": "A", "term": "Myalgia", "synonyms": "muscle pain", "definition": "Pain in muscle", "hierarchy": "", "knowledge_source": "demo"},
        {"code": "B", "term": "Arthralgia", "synonyms": "joint pain", "definition": "Pain in joint", "hierarchy": "", "knowledge_source": "demo"},
    ])
    rows = build_candidate_rationales(
        text="Patient reports severe muscle pain and no joint pain.",
        mention="severe muscle pain",
        candidates=[{"code": "A", "term": "Myalgia", "score": 0.9}, {"code": "B", "term": "Arthralgia", "score": 0.7}],
        terminology=terminology,
        top_k=2,
    )
    assert [row["code"] for row in rows] == ["A", "B"]
    assert all("rationale" in row and "evidence_spans" in row for row in rows)
    assert rows[0]["grounded"] is True
    # Candidate B may be contextually inappropriate; the key invariant here is that
    # its evidence/rationale object is still explicit and auditable rather than omitted.
    assert rows[1]["rationale"]
