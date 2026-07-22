from cohortcoder.review_service import ReviewQueue


def test_review_queue_persists_and_audits(tmp_path):
    q = ReviewQueue(tmp_path / "review.sqlite3")
    assert q.enqueue([{"record_id":"R1","route":"TOP_K_HUMAN_CHOICE","candidate_options":[]}]) == 1
    assert len(q.pending()) == 1
    q.decide("R1", action="SELECT_ALTERNATIVE", selected_code="B", reason="better evidence")
    assert q.pending() == []
    trail = q.audit_trail("R1")
    assert trail[0]["action"] == "SELECT_ALTERNATIVE"
    assert q.summary()["status_counts"]["RESOLVED"] == 1
