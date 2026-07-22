from cohortcoder.audit_replay import build_audit_bundle, validate_audit_bundle, decision_trace


def test_audit_bundle_detects_hash_changes(tmp_path):
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}")
    bundle = tmp_path / "bundle.json"
    build_audit_bundle(bundle, release="0.0.19", files={"metrics": artifact}, decision_semantics={})
    assert validate_audit_bundle(bundle)["valid"] is True
    artifact.write_text('{"changed":true}')
    assert validate_audit_bundle(bundle)["valid"] is False


def test_decision_trace_human_override_is_explicit():
    trace = decision_trace(record_id="R1", model_prediction={"predicted_code":"A"}, uncertainty={}, explanation_quality={}, route_before_review="TOP_K_HUMAN_CHOICE", human_review_event={"action":"SELECT_ALTERNATIVE","selected_code":"B"})
    assert trace["final_code"] == "B"
    assert trace["final_status"] == "HUMAN_ADJUDICATED"
