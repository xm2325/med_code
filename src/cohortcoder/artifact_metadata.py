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
