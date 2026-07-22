from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class ExpertFeedback:
    record_id: str
    model_release: str
    frozen_policy_id: str
    original_predicted_code: str
    original_candidate_codes: tuple[str, ...]
    original_route: str
    human_selected_code: str
    human_action: str
    human_reason: str
    reviewer_id_hash: str = ""
    created_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["original_candidate_codes"] = list(self.original_candidate_codes)
        payload["created_at_utc"] = self.created_at_utc or datetime.now(timezone.utc).isoformat()
        return payload


def hash_reviewer_id(value: str) -> str:
    value = str(value or "").strip()
    return hashlib.sha256(value.encode()).hexdigest()[:16] if value else ""


def validate_feedback(feedback: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    candidates = [str(x) for x in feedback.get("original_candidate_codes", [])]
    selected = str(feedback.get("human_selected_code", ""))
    action = str(feedback.get("human_action", ""))
    if not str(feedback.get("record_id", "")):
        errors.append("missing_record_id")
    if not str(feedback.get("model_release", "")):
        errors.append("missing_model_release")
    if action not in {"ACCEPT_TOP1", "SELECT_ALTERNATIVE", "RECODE_OUTSIDE_TOPK", "ESCALATE", "NO_CODE"}:
        errors.append("invalid_human_action")
    if action in {"ACCEPT_TOP1", "SELECT_ALTERNATIVE"} and selected not in candidates:
        errors.append("selected_code_not_in_original_topk")
    if action == "RECODE_OUTSIDE_TOPK" and (not selected or selected in candidates):
        errors.append("outside_topk_action_inconsistent")
    return errors


def append_feedback_jsonl(path: str | Path, feedback: ExpertFeedback | Mapping[str, Any]) -> None:
    payload = feedback.to_dict() if isinstance(feedback, ExpertFeedback) else dict(feedback)
    errors = validate_feedback(payload)
    if errors:
        raise ValueError("Invalid expert feedback: " + ",".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def feedback_to_training_memory(
    feedback_rows: Iterable[Mapping[str, Any]],
    source_records: pd.DataFrame,
    *,
    minimum_release_exclusive: str | None = None,
) -> pd.DataFrame:
    """Create future-version training-memory rows from adjudicated feedback.

    This function never mutates historical benchmark files. Callers must write the returned
    table to a *new* release-specific memory artifact. TEST labels/results from an earlier
    release remain immutable.
    """
    records = source_records.copy()
    records["record_id"] = records["record_id"].astype(str)
    by_id = {str(row.record_id): row for _, row in records.iterrows()}
    out: list[dict[str, Any]] = []
    for item in feedback_rows:
        action = str(item.get("human_action", ""))
        code = str(item.get("human_selected_code", ""))
        if action in {"ESCALATE", "NO_CODE"} or not code:
            continue
        record_id = str(item.get("record_id", ""))
        if record_id not in by_id:
            continue
        row = by_id[record_id]
        out.append({
            "record_id": record_id,
            "text": str(row.get("text", "")),
            "mention": str(row.get("mention", "")),
            "gold_code": code,
            "gold_term": "",
            "feedback_source_release": str(item.get("model_release", "")),
            "feedback_human_action": action,
            "provenance": "expert_feedback_future_release_only",
        })
    return pd.DataFrame(out)


def feedback_summary(feedback_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(feedback_rows)
    actions: dict[str, int] = {}
    top1_accepts = 0
    alternatives = 0
    outside = 0
    for item in rows:
        action = str(item.get("human_action", ""))
        actions[action] = actions.get(action, 0) + 1
        top1_accepts += action == "ACCEPT_TOP1"
        alternatives += action == "SELECT_ALTERNATIVE"
        outside += action == "RECODE_OUTSIDE_TOPK"
    n = len(rows)
    return {
        "n_feedback": n,
        "action_counts": actions,
        "top1_accept_rate": top1_accepts / n if n else 0.0,
        "topk_rescue_rate": alternatives / n if n else 0.0,
        "outside_topk_rate": outside / n if n else 0.0,
    }
