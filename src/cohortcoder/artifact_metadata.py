from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _patch_json(path: Path, updates: dict[str, Any]) -> None:
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stamp_benchmark_artifacts(
    output_dir: str | Path,
    *,
    version: str,
    benchmark_profile: str | None = None,
) -> None:
    """Stamp user-facing run artifacts with the release that produced the run."""
    output = Path(output_dir)
    common = {"version": str(version)}
    if benchmark_profile:
        common["benchmark_profile"] = str(benchmark_profile)
    for name in ["frozen_policy.json", "experiment_manifest.json"]:
        _patch_json(output / name, common)


def apply_dataset_readiness_gate(
    output_dir: str | Path,
    *,
    dataset_readiness_passed: bool,
    reason: str = "dataset_preflight_audit_failed_or_was_overridden",
) -> None:
    """Prevent a debug audit override from remaining labelled reportable.

    The core Results Contract predates the v0.0.12 dataset-readiness gate. Rather than
    silently changing its historical fields, this function appends the readiness result
    and conservatively downgrades user-facing reportability when the pre-flight audit did
    not pass.
    """
    output = Path(output_dir)
    contract_path = output / "results_contract.json"
    metrics_path = output / "metrics.json"

    contract: dict[str, Any] = {}
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["dataset_readiness_passed"] = bool(dataset_readiness_passed)
    if not dataset_readiness_passed:
        contract["reportable"] = False
        contract["status"] = "non_reportable"
        contract["non_reportable_reason"] = reason
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["dataset_readiness_passed"] = bool(dataset_readiness_passed)
        if not dataset_readiness_passed:
            metrics["results_reportable"] = False
            metrics["results_status"] = "non_reportable"
            metrics["non_reportable_reason"] = reason
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
