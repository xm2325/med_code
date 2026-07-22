from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_SCHEMA_VERSION = "1.0"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_audit_bundle(
    output_path: str | Path,
    *,
    release: str,
    files: Mapping[str, str | Path],
    decision_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_files = {}
    for name, path in files.items():
        p = Path(path)
        manifest_files[name] = {
            "path": str(p),
            "exists": p.exists(),
            "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
        }
    bundle = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "release": release,
        "files": manifest_files,
        "decision_semantics": dict(decision_semantics),
    }
    Path(output_path).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def validate_audit_bundle(path: str | Path, *, verify_files: bool = True) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    failures: list[str] = []
    if str(bundle.get("artifact_schema_version", "")) != ARTIFACT_SCHEMA_VERSION:
        failures.append("unsupported_artifact_schema_version")
    for name, info in (bundle.get("files") or {}).items():
        if not info.get("exists"):
            failures.append(f"missing_artifact:{name}")
            continue
        if verify_files:
            p = Path(str(info.get("path", "")))
            if not p.exists():
                failures.append(f"artifact_no_longer_exists:{name}")
            elif info.get("sha256") and sha256_file(p) != info.get("sha256"):
                failures.append(f"artifact_hash_mismatch:{name}")
    return {"valid": not failures, "failures": failures, "release": bundle.get("release"), "schema": bundle.get("artifact_schema_version")}


def decision_trace(
    *,
    record_id: str,
    model_prediction: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    explanation_quality: Mapping[str, Any],
    route_before_review: str,
    human_review_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    final_code = str(model_prediction.get("predicted_code", ""))
    final_status = route_before_review
    if human_review_event:
        action = str(human_review_event.get("action", ""))
        selected = str(human_review_event.get("selected_code", ""))
        if action in {"ACCEPT_TOP1", "SELECT_ALTERNATIVE", "RECODE_OUTSIDE_TOPK"} and selected:
            final_code = selected
            final_status = "HUMAN_ADJUDICATED"
        elif action == "NO_CODE":
            final_code = ""
            final_status = "HUMAN_ADJUDICATED_NO_CODE"
        elif action == "ESCALATE":
            final_status = "ESCALATED"
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_id": str(record_id),
        "model_prediction": dict(model_prediction),
        "uncertainty": dict(uncertainty),
        "explanation_quality": dict(explanation_quality),
        "route_before_review": str(route_before_review),
        "human_review_event": dict(human_review_event or {}),
        "final_code": final_code,
        "final_status": final_status,
    }
