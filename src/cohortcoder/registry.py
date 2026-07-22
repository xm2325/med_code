from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def register_experiment(
    registry_path: str | Path,
    *,
    release: str,
    benchmark_profile: str,
    data_fingerprints: Mapping[str, Any],
    model_policy: Mapping[str, Any],
    metrics: Mapping[str, Any],
    results_contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "release": release,
        "benchmark_profile": benchmark_profile,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_fingerprints": dict(data_fingerprints),
        "model_policy": dict(model_policy),
        "metrics": dict(metrics),
        "results_contract": dict(results_contract),
    }
    payload["experiment_id"] = stable_json_hash({k: v for k, v in payload.items() if k != "created_at_utc"})[:16]
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def write_data_card(path: str | Path, *, name: str, source: str, licence: str, task: str, split_unit: str, governance: str, limitations: list[str]) -> None:
    text = f"""# Data card — {name}\n\n- **Source:** {source}\n- **Licence/access:** {licence}\n- **Task:** {task}\n- **Split unit:** {split_unit}\n- **Governance:** {governance}\n\n## Known limitations\n\n""" + "\n".join(f"- {item}" for item in limitations) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def write_model_card(path: str | Path, *, release: str, intended_use: str, model_components: list[str], safety_boundaries: list[str], evaluation: Mapping[str, Any]) -> None:
    text = f"# Model card — MedCode {release}\n\n## Intended use\n\n{intended_use}\n\n## Components\n\n"
    text += "\n".join(f"- {item}" for item in model_components)
    text += "\n\n## Safety boundaries\n\n" + "\n".join(f"- {item}" for item in safety_boundaries)
    text += "\n\n## Evaluation snapshot\n\n```json\n" + json.dumps(dict(evaluation), indent=2) + "\n```\n"
    Path(path).write_text(text, encoding="utf-8")
