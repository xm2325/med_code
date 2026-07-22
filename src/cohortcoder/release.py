from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


V010_REQUIRED_CAPABILITIES = {
    "audited_real_data_adapter",
    "held_out_evaluation",
    "results_contract",
    "frozen_policy",
    "uncertainty_routing",
    "topk_grounded_rationales",
    "persistent_human_review",
    "expert_feedback_ledger",
    "audit_replay",
}


def build_release_manifest(*, release: str, capabilities: Mapping[str, bool], notes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    missing = sorted(name for name in V010_REQUIRED_CAPABILITIES if not bool(capabilities.get(name, False)))
    return {
        "release": release,
        "required_capabilities": sorted(V010_REQUIRED_CAPABILITIES),
        "capabilities": dict(capabilities),
        "missing_required_capabilities": missing,
        "software_release_complete": not missing,
        "notes": dict(notes or {}),
        "important_boundary": "software_release_complete does not imply clinical deployment readiness or proven real-data performance",
    }


def validate_release_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    if str(manifest.get("release", "")) != "0.1.0":
        failures.append("unexpected_release_version")
    if not bool(manifest.get("software_release_complete", False)):
        failures.append("required_capabilities_missing")
    missing = list(manifest.get("missing_required_capabilities", []))
    if missing:
        failures.extend(f"missing:{item}" for item in missing)
    return {"valid": not failures, "failures": failures}


def write_release_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
