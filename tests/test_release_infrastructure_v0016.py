from cohortcoder.release_gate import evaluate_release_readiness


def test_release_gate_blocks_non_reportable_or_missing_artifacts():
    result = evaluate_release_readiness(
        results_contract={"reportable": False},
        required_artifacts_present={"metrics": True, "frozen_policy": False},
    )
    assert result["release_ready"] is False
    assert "results_not_reportable" in result["failures"]


def test_release_gate_can_pass_complete_candidate():
    result = evaluate_release_readiness(
        results_contract={"reportable": True},
        dataset_audit={"ready_for_benchmark": True},
        explanation_quality={"fail_rate": 0.05},
        required_artifacts_present={"metrics": True, "frozen_policy": True, "results_contract": True},
    )
    assert result["release_ready"] is True
