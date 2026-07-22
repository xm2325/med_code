from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .multilabel import parse_code_list


def _quantiles(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {}
    q = values.astype(float).quantile([0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {str(index): float(value) for index, value in q.items()}


def audit_mimic_records(
    records: pd.DataFrame,
    terminology: pd.DataFrame,
    *,
    minimum_terminology_coverage: float = 0.99,
) -> dict[str, Any]:
    required = {"record_id", "subject_id", "text", "gold_codes_json", "split"}
    if not required.issubset(records.columns):
        raise ValueError(f"MIMIC records require {sorted(required)}")
    df = records.copy().fillna("")
    split_subjects = {
        split: set(group["subject_id"].astype(str))
        for split, group in df.groupby("split")
    }
    overlaps = {
        "train_val": len(split_subjects.get("train", set()) & split_subjects.get("val", set())),
        "train_test": len(split_subjects.get("train", set()) & split_subjects.get("test", set())),
        "val_test": len(split_subjects.get("val", set()) & split_subjects.get("test", set())),
    }

    code_lists = df["gold_codes_json"].map(parse_code_list)
    all_codes = sorted({code for values in code_lists for code in values})
    known = set(terminology["code"].astype(str))
    missing = sorted(set(all_codes) - known)
    coverage = 1.0 - (len(missing) / len(all_codes) if all_codes else 0.0)

    text_lengths = df["text"].astype(str).str.len()
    code_counts = code_lists.map(len)
    empty_text = int((df["text"].astype(str).str.strip() == "").sum())
    empty_labels = int((code_counts == 0).sum())

    hard_failures: list[str] = []
    warnings: list[str] = []
    if any(overlaps.values()):
        hard_failures.append("subject_id_leakage_across_splits")
    if empty_text:
        hard_failures.append("empty_clinical_text")
    if empty_labels:
        hard_failures.append("records_without_icd10_labels")
    if coverage < minimum_terminology_coverage:
        hard_failures.append("terminology_coverage_below_threshold")
    if len(all_codes) < 10:
        warnings.append("very_small_code_vocabulary")

    split_counts = {
        str(split): {
            "records": int(len(group)),
            "subjects": int(group["subject_id"].astype(str).nunique()),
        }
        for split, group in df.groupby("split")
    }
    return {
        "dataset": "MIMIC-IV-Note -> ICD-10",
        "records": int(len(df)),
        "subjects": int(df["subject_id"].astype(str).nunique()),
        "unique_icd10_codes": int(len(all_codes)),
        "split_counts": split_counts,
        "subject_overlap": overlaps,
        "empty_text_records": empty_text,
        "records_without_labels": empty_labels,
        "terminology_coverage": float(coverage),
        "minimum_terminology_coverage": float(minimum_terminology_coverage),
        "missing_terminology_codes": missing[:200],
        "note_length_char_quantiles": _quantiles(text_lengths),
        "codes_per_note_quantiles": _quantiles(code_counts),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "ready_for_benchmark": not hard_failures,
    }


def make_mimic_review_sample(records: pd.DataFrame, n: int = 30, seed: int = 42) -> pd.DataFrame:
    df = records.copy().fillna("")
    df["_text_len"] = df["text"].astype(str).str.len()
    df["_code_count"] = df["gold_codes_json"].map(lambda value: len(parse_code_list(value)))
    # Deliberately include operational extremes, then random controls.
    extremes = pd.concat([
        df.nlargest(min(5, len(df)), "_text_len"),
        df.nlargest(min(5, len(df)), "_code_count"),
    ]).drop_duplicates("record_id")
    remaining = max(0, int(n) - len(extremes))
    pool = df[~df["record_id"].isin(extremes["record_id"])]
    if remaining and len(pool) > remaining:
        pool = pool.sample(n=remaining, random_state=seed)
    elif remaining == 0:
        pool = pool.head(0)
    selected = pd.concat([extremes, pool], ignore_index=True).head(int(n)).copy()
    selected = selected.drop(columns=["_text_len", "_code_count"], errors="ignore")
    for column in ["review_note_alignment", "review_label_set_plausible", "review_comments"]:
        selected[column] = ""
    return selected


def write_mimic_audit_artifacts(
    records: pd.DataFrame,
    terminology: pd.DataFrame,
    output_dir: str | Path,
    *,
    sample_size: int = 30,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_mimic_records(records, terminology)
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    make_mimic_review_sample(records, n=sample_size).to_csv(output / "manual_data_review_sample.csv", index=False)
    return audit
