import pandas as pd

from cohortcoder.feedback import feedback_summary, feedback_to_training_memory, validate_feedback


def test_feedback_requires_consistent_topk_selection():
    good = {"record_id":"R1","model_release":"0.0.15","original_candidate_codes":["A","B"],"human_selected_code":"B","human_action":"SELECT_ALTERNATIVE"}
    assert validate_feedback(good) == []
    bad = dict(good, human_selected_code="C")
    assert "selected_code_not_in_original_topk" in validate_feedback(bad)


def test_feedback_memory_is_new_artifact_not_in_place_mutation():
    source = pd.DataFrame([{"record_id":"R1","text":"muscle pain","mention":"muscle pain"}])
    rows = [{"record_id":"R1","model_release":"0.0.15","human_selected_code":"A","human_action":"ACCEPT_TOP1"}]
    memory = feedback_to_training_memory(rows, source)
    assert memory.iloc[0].gold_code == "A"
    assert "gold_code" not in source.columns
    summary = feedback_summary(rows)
    assert summary["top1_accept_rate"] == 1.0
