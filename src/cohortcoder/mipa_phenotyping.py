"""Local/offline evaluation utilities for evidence-grounded MIPA phenotyping.

This module deliberately performs no network or external-API calls. It is designed for
credentialed MIMIC/MIPA environments where discharge-summary text must remain inside an
approved compute boundary.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

DEFAULT_PHENOTYPES = (
    "hypertension",
    "depression",
    "diabetes_type_2",
    "hfpef",
    "vte_past",
    "obesity",
)

POSITIVE_VALUES = {"1", "true", "yes", "positive", "present"}
NEGATIVE_VALUES = {"0", "false", "no", "negative", "absent"}
ASSERTION_VALUES = {"present", "absent", "possible", "negated", "family_history", "unknown"}
TEMPORALITY_VALUES = {"current", "historical", "resolved", "unknown"}


@dataclass(frozen=True)
class AcceptanceThresholds:
    macro_f1: float = 0.85
    phenotype_f1: float = 0.80
    min_phenotypes_passing: int = 5
    common_phenotype_recall: float = 0.60
    common_positive_min: int = 20
    evidence_verbatim_rate: float = 0.99
    evidence_support_rate: float = 0.90
    severe_context_error_rate: float = 0.05


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _read_predictions(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Prediction JSONL line {line_number} is not an object")
                rows.append(payload)
        if not rows:
            raise ValueError(f"Prediction JSONL is empty: {path}")
        return rows
    return [dict(row) for row in _read_csv(path)]


def _binary(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized in POSITIVE_VALUES:
        return 1
    if normalized in NEGATIVE_VALUES:
        return 0
    raise ValueError(f"Unsupported binary value: {value!r}")


def _require_columns(rows: Sequence[Mapping[str, object]], columns: Iterable[str], label: str) -> None:
    present = set(rows[0]) if rows else set()
    missing = [column for column in columns if column not in present]
    if missing:
        raise ValueError(f"{label} schema missing columns: {missing}")


def detect_join_key(labels: Sequence[Mapping[str, object]], notes: Sequence[Mapping[str, object]]) -> str:
    for key in ("note_id", "hadm_id"):
        if key in labels[0] and key in notes[0]:
            return key
    raise ValueError("Labels and notes must share note_id or hadm_id")


def detect_text_column(notes: Sequence[Mapping[str, object]]) -> str:
    for column in ("text", "note_text", "discharge_summary"):
        if column in notes[0]:
            return column
    raise ValueError("Notes schema must include one of: text, note_text, discharge_summary")


def assign_subject_split(
    subject_id: object,
    *,
    seed: str = "20260723",
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> str:
    if not 0 <= train_fraction <= 1 or not 0 <= validation_fraction <= 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    digest = hashlib.sha256(f"{seed}|{subject_id}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(16**16)
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def build_subject_split_manifest(
    labels: Sequence[Mapping[str, object]],
    *,
    seed: str = "20260723",
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> list[dict[str, str]]:
    _require_columns(labels, ["subject_id"], "MIPA labels")
    subject_to_split: dict[str, str] = {}
    manifest: list[dict[str, str]] = []
    for row in labels:
        subject_id = str(row["subject_id"])
        split = assign_subject_split(
            subject_id,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        previous = subject_to_split.setdefault(subject_id, split)
        if previous != split:
            raise AssertionError("Subject assigned to multiple splits")
        manifest.append(
            {
                "note_id": str(row.get("note_id", "")),
                "hadm_id": str(row.get("hadm_id", "")),
                "subject_id": subject_id,
                "split": split,
            }
        )
    return manifest


def split_audit(manifest: Sequence[Mapping[str, str]]) -> dict[str, object]:
    split_subjects: dict[str, set[str]] = defaultdict(set)
    split_rows = Counter()
    for row in manifest:
        split = row["split"]
        split_rows[split] += 1
        split_subjects[split].add(row["subject_id"])
    intersections: dict[str, int] = {}
    names = sorted(split_subjects)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            intersections[f"{left}__{right}"] = len(split_subjects[left] & split_subjects[right])
    return {
        "rows_per_split": dict(sorted(split_rows.items())),
        "subjects_per_split": {key: len(value) for key, value in sorted(split_subjects.items())},
        "subject_overlap_counts": intersections,
        "subject_disjoint": all(value == 0 for value in intersections.values()),
    }


def _metric_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int | None]:
    counts = _metric_counts(y_true, y_pred)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    if precision is None or recall is None or precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        **counts,
        "n": len(y_true),
        "gold_positive": sum(y_true),
        "predicted_positive": sum(y_pred),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def _normalise_optional(value: object, allowed: set[str], field: str) -> tuple[str, bool]:
    normalized = str(value or "unknown").strip().lower() or "unknown"
    return normalized, normalized in allowed


def _load_evidence_audit(path: str | Path | None) -> dict[tuple[str, str], dict[str, int]]:
    if path is None:
        return {}
    rows = _read_csv(path)
    _require_columns(rows, ["note_id", "phenotype", "supports_prediction", "severe_context_error"], "Evidence audit")
    output: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (str(row["note_id"]), str(row["phenotype"]))
        output[key] = {
            "supports_prediction": _binary(row["supports_prediction"]),
            "severe_context_error": _binary(row["severe_context_error"]),
        }
    return output


def evaluate_mipa_predictions(
    *,
    labels_path: str | Path,
    notes_path: str | Path,
    predictions_path: str | Path,
    phenotypes: Sequence[str] = DEFAULT_PHENOTYPES,
    evaluation_scope: str = "all",
    split_seed: str = "20260723",
    evidence_audit_path: str | Path | None = None,
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    """Evaluate local predictions without transmitting note text outside the local process.

    Returns summary, per-phenotype metrics, error rows, and a subject-disjoint split manifest.
    MIPA public labels supervise phenotype presence only. Assertion and temporality are
    contract-validated but are not scored as clinical accuracy unless separately annotated.
    """
    labels = _read_csv(labels_path)
    notes = _read_csv(notes_path)
    predictions = _read_predictions(predictions_path)
    _require_columns(labels, ["note_id", "subject_id", *phenotypes], "MIPA labels")
    _require_columns(predictions, ["note_id", "phenotype", "prediction", "evidence", "assertion", "temporality"], "Predictions")

    join_key = detect_join_key(labels, notes)
    text_column = detect_text_column(notes)
    manifest = build_subject_split_manifest(labels, seed=split_seed)
    audit = split_audit(manifest)
    if not audit["subject_disjoint"]:
        raise AssertionError("Subject leakage detected across split manifest")

    split_by_note = {row["note_id"]: row["split"] for row in manifest}
    labels_by_note = {str(row["note_id"]): row for row in labels}
    notes_by_join = {str(row[join_key]): row for row in notes}
    label_join_value = {str(row["note_id"]): str(row[join_key]) for row in labels}
    evidence_audit = _load_evidence_audit(evidence_audit_path)

    duplicate_prediction_keys = Counter((str(row["note_id"]), str(row["phenotype"])) for row in predictions)
    duplicated = [key for key, count in duplicate_prediction_keys.items() if count > 1]
    if duplicated:
        raise ValueError(f"Duplicate prediction keys found, e.g. {duplicated[:3]}")

    prediction_by_key = {(str(row["note_id"]), str(row["phenotype"])): row for row in predictions}
    if evaluation_scope not in {"all", "train", "validation", "test"}:
        raise ValueError("evaluation_scope must be one of all/train/validation/test")

    per_phenotype: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    evidence_positive_total = 0
    evidence_positive_verbatim = 0
    contract_invalid = 0
    expected_predictions = 0
    observed_predictions = 0
    audited_support: list[int] = []
    audited_context_errors: list[int] = []

    for phenotype in phenotypes:
        y_true: list[int] = []
        y_pred: list[int] = []
        for note_id, label_row in labels_by_note.items():
            split = split_by_note[note_id]
            if evaluation_scope != "all" and split != evaluation_scope:
                continue
            expected_predictions += 1
            key = (note_id, phenotype)
            prediction_row = prediction_by_key.get(key)
            if prediction_row is None:
                error_rows.append({
                    "note_id": note_id,
                    "subject_id": str(label_row["subject_id"]),
                    "split": split,
                    "phenotype": phenotype,
                    "error_type": "missing_prediction",
                    "gold": _binary(label_row[phenotype]),
                    "prediction": "",
                    "evidence_verbatim": "",
                    "assertion_valid": "",
                    "temporality_valid": "",
                })
                continue
            observed_predictions += 1
            gold = _binary(label_row[phenotype])
            try:
                pred = _binary(prediction_row["prediction"])
            except ValueError:
                contract_invalid += 1
                error_rows.append({
                    "note_id": note_id,
                    "subject_id": str(label_row["subject_id"]),
                    "split": split,
                    "phenotype": phenotype,
                    "error_type": "invalid_prediction_value",
                    "gold": gold,
                    "prediction": str(prediction_row["prediction"]),
                    "evidence_verbatim": "",
                    "assertion_valid": "",
                    "temporality_valid": "",
                })
                continue

            assertion, assertion_valid = _normalise_optional(prediction_row.get("assertion"), ASSERTION_VALUES, "assertion")
            temporality, temporality_valid = _normalise_optional(prediction_row.get("temporality"), TEMPORALITY_VALUES, "temporality")
            if not assertion_valid or not temporality_valid:
                contract_invalid += 1

            note_row = notes_by_join.get(label_join_value[note_id])
            if note_row is None:
                raise ValueError(f"No note text found for {join_key}={label_join_value[note_id]}")
            note_text = str(note_row[text_column])
            evidence = str(prediction_row.get("evidence", "") or "").strip()
            evidence_verbatim = bool(evidence) and evidence in note_text
            if pred == 1:
                evidence_positive_total += 1
                evidence_positive_verbatim += int(evidence_verbatim)

            audit_row = evidence_audit.get(key)
            if audit_row is not None:
                audited_support.append(audit_row["supports_prediction"])
                audited_context_errors.append(audit_row["severe_context_error"])

            y_true.append(gold)
            y_pred.append(pred)
            if gold != pred or (pred == 1 and not evidence_verbatim) or not assertion_valid or not temporality_valid:
                reasons = []
                if gold != pred:
                    reasons.append("classification_error")
                if pred == 1 and not evidence_verbatim:
                    reasons.append("non_verbatim_or_missing_evidence")
                if not assertion_valid:
                    reasons.append("invalid_assertion")
                if not temporality_valid:
                    reasons.append("invalid_temporality")
                error_rows.append({
                    "note_id": note_id,
                    "subject_id": str(label_row["subject_id"]),
                    "split": split,
                    "phenotype": phenotype,
                    "error_type": ";".join(reasons),
                    "gold": gold,
                    "prediction": pred,
                    "evidence_verbatim": evidence_verbatim,
                    "assertion_valid": assertion_valid,
                    "temporality_valid": temporality_valid,
                })

        metrics = _metrics(y_true, y_pred)
        metrics["phenotype"] = phenotype
        per_phenotype.append(metrics)

    f1_values = [float(row["f1"]) for row in per_phenotype if row["n"]]
    recall_failures = [
        str(row["phenotype"])
        for row in per_phenotype
        if int(row["gold_positive"] or 0) >= thresholds.common_positive_min
        and row["recall"] is not None
        and float(row["recall"]) < thresholds.common_phenotype_recall
    ]
    min_required = min(thresholds.min_phenotypes_passing, len(phenotypes))
    n_f1_passing = sum(float(row["f1"]) >= thresholds.phenotype_f1 for row in per_phenotype)
    macro_f1 = mean(f1_values) if f1_values else 0.0
    evidence_verbatim_rate = _safe_ratio(evidence_positive_verbatim, evidence_positive_total)
    coverage = _safe_ratio(observed_predictions, expected_predictions)
    evidence_support_rate = mean(audited_support) if audited_support else None
    severe_context_error_rate = mean(audited_context_errors) if audited_context_errors else None

    automated_gate_checks = {
        "macro_f1": macro_f1 >= thresholds.macro_f1,
        "phenotype_f1_count": n_f1_passing >= min_required,
        "common_phenotype_recall": not recall_failures,
        "prediction_coverage": coverage == 1.0,
        "contract_validity": contract_invalid == 0,
        "evidence_verbatim": evidence_verbatim_rate is not None and evidence_verbatim_rate >= thresholds.evidence_verbatim_rate,
        "subject_disjoint": bool(audit["subject_disjoint"]),
    }
    automated_gate_pass = all(automated_gate_checks.values())

    human_gate_available = bool(audited_support)
    human_gate_checks = {
        "evidence_support": evidence_support_rate is not None and evidence_support_rate >= thresholds.evidence_support_rate,
        "severe_context_error": severe_context_error_rate is not None and severe_context_error_rate < thresholds.severe_context_error_rate,
    }
    human_gate_pass = human_gate_available and all(human_gate_checks.values())
    if not automated_gate_pass:
        final_status = "FAIL_AUTOMATED_GATE"
    elif not human_gate_available:
        final_status = "PASS_AUTOMATED_PENDING_HUMAN_EVIDENCE_AUDIT"
    elif human_gate_pass:
        final_status = "PASS"
    else:
        final_status = "FAIL_HUMAN_EVIDENCE_GATE"

    summary: dict[str, object] = {
        "schema_version": "mipa-local-phenotyping-v0.3.0",
        "governance": {
            "network_calls_performed": False,
            "external_api_calls_performed": False,
            "restricted_note_text_written_to_summary": False,
            "design": "local/offline evaluation only",
        },
        "input": {
            "join_key": join_key,
            "text_column": text_column,
            "phenotypes": list(phenotypes),
            "evaluation_scope": evaluation_scope,
        },
        "split_audit": audit,
        "performance": {
            "macro_f1": macro_f1,
            "n_phenotypes_f1_at_or_above_threshold": n_f1_passing,
            "min_phenotypes_required": min_required,
            "common_phenotype_recall_failures": recall_failures,
            "prediction_coverage": coverage,
        },
        "evidence_contract": {
            "positive_predictions": evidence_positive_total,
            "positive_predictions_with_verbatim_evidence": evidence_positive_verbatim,
            "verbatim_evidence_rate": evidence_verbatim_rate,
            "invalid_contract_records": contract_invalid,
            "assertion_temporality_accuracy_scored": False,
            "reason": "MIPA public gold labels supervise phenotype presence, not assertion or temporality.",
        },
        "human_evidence_audit": {
            "available": human_gate_available,
            "n_audited": len(audited_support),
            "evidence_support_rate": evidence_support_rate,
            "severe_context_error_rate": severe_context_error_rate,
        },
        "acceptance": {
            "thresholds": thresholds.__dict__,
            "automated_gate_checks": automated_gate_checks,
            "automated_gate_pass": automated_gate_pass,
            "human_gate_checks": human_gate_checks,
            "human_gate_pass": human_gate_pass,
            "final_status": final_status,
        },
    }
    return summary, per_phenotype, error_rows, manifest


def write_evaluation_outputs(
    output_dir: str | Path,
    summary: Mapping[str, object],
    per_phenotype: Sequence[Mapping[str, object]],
    errors: Sequence[Mapping[str, object]],
    manifest: Sequence[Mapping[str, str]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_csv(name: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    metric_fields = [
        "phenotype", "n", "gold_positive", "predicted_positive", "tp", "tn", "fp", "fn",
        "precision", "recall", "specificity", "f1",
    ]
    write_csv("phenotype_metrics.csv", per_phenotype, metric_fields)
    error_fields = [
        "note_id", "subject_id", "split", "phenotype", "error_type", "gold", "prediction",
        "evidence_verbatim", "assertion_valid", "temporality_valid",
    ]
    write_csv("error_cases.csv", errors, error_fields)
    write_csv("subject_split_manifest.csv", manifest, ["note_id", "hadm_id", "subject_id", "split"])
