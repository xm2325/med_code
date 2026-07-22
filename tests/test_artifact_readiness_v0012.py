import json

from cohortcoder.artifact_metadata import apply_dataset_readiness_gate


def test_failed_dataset_audit_downgrades_reportability(tmp_path):
    (tmp_path / "results_contract.json").write_text(json.dumps({"reportable": True, "status": "reportable"}), encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"results_reportable": True, "results_status": "reportable"}), encoding="utf-8")

    apply_dataset_readiness_gate(tmp_path, dataset_readiness_passed=False)

    contract = json.loads((tmp_path / "results_contract.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert contract["reportable"] is False
    assert contract["status"] == "non_reportable"
    assert contract["dataset_readiness_passed"] is False
    assert metrics["results_reportable"] is False
    assert metrics["results_status"] == "non_reportable"
