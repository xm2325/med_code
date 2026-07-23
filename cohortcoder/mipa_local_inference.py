"""Local command-based inference for evidence-grounded MIPA phenotyping.

The MedCode harness itself performs no HTTP/API calls. Clinical note text is passed only to a
user-supplied local child process over stdin. The approved execution environment remains
responsible for enforcing host/network isolation for that child process.
"""
from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .mipa_phenotyping import (
    ASSERTION_VALUES,
    DEFAULT_PHENOTYPES,
    TEMPORALITY_VALUES,
    _binary,
    _read_predictions,
    _require_columns,
    detect_text_column,
)

PROMPT_VERSION = "mipa-evidence-grounded-v0.3.0"

PHENOTYPE_DISPLAY_NAMES = {
    "hypertension": "Hypertension",
    "depression": "Depression",
    "diabetes_type_2": "Type 2 diabetes mellitus",
    "hfpef": "Heart failure with preserved ejection fraction (HFpEF)",
    "vte_past": "Previous venous thromboembolism (VTE)",
    "obesity": "Obesity",
}


@dataclass(frozen=True)
class InferenceConfig:
    command: tuple[str, ...]
    timeout_seconds: float = 180.0
    model_id: str = "local-model"
    resume: bool = True


def build_prompt_payload(*, note_id: str, phenotype: str, note_text: str) -> dict[str, object]:
    """Build the exact request sent to the local model process.

    Gold labels are intentionally absent from this payload to prevent label leakage.
    """
    display_name = PHENOTYPE_DISPLAY_NAMES.get(phenotype, phenotype)
    system = (
        "You are performing clinical phenotype identification from one discharge summary. "
        "Use only evidence explicitly supported by the note. Return exactly one JSON object and no markdown. "
        "For a positive prediction, copy a short exact quote from the note into evidence. "
        "Do not treat negation, family history, or mere diagnostic possibility as a confirmed patient phenotype."
    )
    user = (
        f"Target phenotype: {display_name}\n\n"
        "Return JSON with exactly these fields:\n"
        '{"prediction": 0 or 1, "evidence": "exact quote or empty string", '
        '"assertion": "present|absent|possible|negated|family_history|unknown", '
        '"temporality": "current|historical|resolved|unknown", "confidence": 0.0 to 1.0}\n\n'
        "Clinical note:\n"
        f"{note_text}"
    )
    return {
        "schema_version": "mipa-local-inference-request-v0.3.0",
        "prompt_version": PROMPT_VERSION,
        "note_id": str(note_id),
        "phenotype": phenotype,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def parse_model_response(text: str) -> dict[str, object]:
    """Parse and strictly validate one model JSON response."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Local model returned an empty response")
    if stripped.startswith("```"):
        raise ValueError("Model response must be raw JSON without markdown fences")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Model response must be one JSON object")

    required = {"prediction", "evidence", "assertion", "temporality", "confidence"}
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    if missing:
        raise ValueError(f"Model response missing fields: {missing}")
    if extra:
        raise ValueError(f"Model response has unexpected fields: {extra}")

    prediction = _binary(payload["prediction"])
    evidence = payload["evidence"]
    if not isinstance(evidence, str):
        raise ValueError("evidence must be a string")

    assertion = str(payload["assertion"]).strip().lower()
    temporality = str(payload["temporality"]).strip().lower()
    if assertion not in ASSERTION_VALUES:
        raise ValueError(f"Unsupported assertion value: {assertion!r}")
    if temporality not in TEMPORALITY_VALUES:
        raise ValueError(f"Unsupported temporality value: {temporality!r}")

    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    if prediction == 1 and not evidence.strip():
        raise ValueError("Positive predictions must include non-empty evidence")

    return {
        "prediction": prediction,
        "evidence": evidence.strip(),
        "assertion": assertion,
        "temporality": temporality,
        "confidence": confidence,
    }


def run_local_command(
    command: Sequence[str],
    payload: Mapping[str, object],
    *,
    timeout_seconds: float,
) -> str:
    """Run one local child process without a shell and return stdout.

    The harness sets common ML-library offline flags, but operating-system/container network
    isolation must still be enforced by the approved environment.
    """
    if not command:
        raise ValueError("Local backend command is empty")
    joined = " ".join(command).lower()
    if "http://" in joined or "https://" in joined:
        raise ValueError("Remote URL-like backend commands are not permitted")

    env = os.environ.copy()
    env.update(
        {
            "MEDCODE_LOCAL_ONLY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        list(command),
        input=json.dumps(dict(payload), ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        shell=False,
        env=env,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"Local backend exited with code {completed.returncode}: {stderr[:500]}"
        )
    return completed.stdout


def parse_command(command: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, str):
        parsed = tuple(shlex.split(command))
    else:
        parsed = tuple(str(part) for part in command)
    if not parsed:
        raise ValueError("Local backend command is empty")
    return parsed


def _read_notes(path: str | Path) -> tuple[list[dict[str, str]], str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Notes CSV is empty: {path}")
    _require_columns(rows, ["note_id"], "MIPA notes")
    text_column = detect_text_column(rows)
    note_ids = [str(row["note_id"]) for row in rows]
    if len(set(note_ids)) != len(note_ids):
        raise ValueError("Notes contain duplicate note_id values")
    return rows, text_column


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    rows = _read_predictions(path)
    _require_columns(rows, ["note_id", "phenotype"], "Existing predictions")
    keys = [(str(row["note_id"]), str(row["phenotype"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Existing predictions contain duplicate note_id/phenotype keys")
    return set(keys)


def generate_local_predictions(
    *,
    notes_path: str | Path,
    predictions_path: str | Path,
    failures_path: str | Path,
    config: InferenceConfig,
    phenotypes: Sequence[str] = DEFAULT_PHENOTYPES,
    limit_notes: int | None = None,
) -> dict[str, object]:
    """Generate checkpointed predictions through a local command backend.

    Output JSONL never contains full note text. Each completed note/phenotype pair is appended
    immediately so an interrupted run can resume without repeating finished work.
    """
    notes, text_column = _read_notes(notes_path)
    if limit_notes is not None:
        if limit_notes < 1:
            raise ValueError("limit_notes must be >= 1")
        notes = notes[:limit_notes]

    predictions_path = Path(predictions_path)
    failures_path = Path(failures_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _existing_keys(predictions_path) if config.resume else set()
    if not config.resume:
        predictions_path.write_text("", encoding="utf-8")
        failures_path.write_text("", encoding="utf-8")

    expected_pairs = len(notes) * len(phenotypes)
    generated = 0
    skipped_existing = 0
    failures = 0
    evidence_verbatim_failures = 0

    for note in notes:
        note_id = str(note["note_id"])
        note_text = str(note[text_column])
        for phenotype in phenotypes:
            key = (note_id, phenotype)
            if key in existing:
                skipped_existing += 1
                continue
            request = build_prompt_payload(note_id=note_id, phenotype=phenotype, note_text=note_text)
            try:
                stdout = run_local_command(
                    config.command,
                    request,
                    timeout_seconds=config.timeout_seconds,
                )
                parsed = parse_model_response(stdout)
                evidence = str(parsed["evidence"])
                evidence_verbatim = (not evidence) or evidence in note_text
                if parsed["prediction"] == 1 and not evidence_verbatim:
                    evidence_verbatim_failures += 1
                row = {
                    "note_id": note_id,
                    "phenotype": phenotype,
                    **parsed,
                    "evidence_verbatim_at_generation": evidence_verbatim,
                    "model_id": config.model_id,
                    "prompt_version": PROMPT_VERSION,
                }
                with predictions_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                generated += 1
                existing.add(key)
            except Exception as exc:  # continue to preserve progress; acceptance catches missing pairs
                failures += 1
                failure = {
                    "note_id": note_id,
                    "phenotype": phenotype,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "model_id": config.model_id,
                    "prompt_version": PROMPT_VERSION,
                }
                with failures_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")

    completed_pairs = len(existing & {
        (str(note["note_id"]), phenotype)
        for note in notes
        for phenotype in phenotypes
    })
    summary = {
        "schema_version": "mipa-local-inference-summary-v0.3.0",
        "prompt_version": PROMPT_VERSION,
        "model_id": config.model_id,
        "governance": {
            "harness_http_calls": False,
            "harness_external_api_calls": False,
            "note_transport": "stdin to user-supplied local child process",
            "offline_environment_flags_set": True,
            "os_or_container_network_isolation_verified_by_harness": False,
            "network_isolation_requirement": (
                "The approved host/container must enforce network isolation for the child model process."
            ),
            "full_note_text_written_to_prediction_outputs": False,
        },
        "counts": {
            "notes": len(notes),
            "phenotypes": len(phenotypes),
            "expected_pairs": expected_pairs,
            "generated_this_run": generated,
            "skipped_existing": skipped_existing,
            "completed_pairs": completed_pairs,
            "failures_this_run": failures,
            "positive_nonverbatim_evidence_this_run": evidence_verbatim_failures,
        },
        "complete": completed_pairs == expected_pairs and failures == 0,
    }
    return summary
