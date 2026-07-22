from __future__ import annotations

from typing import Any, Mapping


def evaluate_release_readiness(
    *,
    results_contract: Mapping[str, Any],
    dataset_audit: Mapping[str, Any] | None = None,
    explanation_quality: Mapping[str, Any] | None = None,
    required_artifacts_present: Mapping[str, bool] | None = None,
    max_explanation_fail_rate: float = 0.10,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if not bool(results_contract.get("reportable", False)):
        failures.append("results_not_reportable")
    audit = dataset_audit or {}
    if audit and audit.get("ready_for_benchmark") is False:
        failures.append("dataset_audit_failed")
    quality = explanation_quality or {}
    if quality:
        fail_rate = quality.get("fail_rate")
        if fail_rate is not None and float(fail_rate) > float(max_explanation_fail_rate):
            failures.append("explanation_fail_rate_above_release_limit")
    artifacts = required_artifacts_present or {}
    missing = [name for name, present in artifacts.items() if not present]
    if missing:
        failures.append("missing_required_artifacts:" + ",".join(sorted(missing)))
    if not quality:
        warnings.append("explanation_quality_not_evaluated")
    return {
        "release_ready": not failures,
        "failures": failures,
        "warnings": warnings,
        "rule": "all hard release gates must pass; warnings do not promote a failed release",
    }
