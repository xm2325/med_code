from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


_TEXTBOUND_RE = re.compile(r"^(T\d+)\t([^\t]+)\t(.*)$")
_CODE_RE = re.compile(r"(?:MedDRA\s*[:#]?\s*)?(\d{6,9})\b", re.I)
_TARGET_RE = re.compile(r"\b(T\d+)\b")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _parse_textbound(line: str, source_text: str) -> dict[str, Any] | None:
    match = _TEXTBOUND_RE.match(line)
    if not match:
        return None
    annotation_id, meta, mention = match.groups()
    if " " not in meta:
        return None
    entity_type, span_spec = meta.split(" ", 1)
    spans: list[tuple[int, int]] = []
    for segment in span_spec.split(";"):
        nums = re.findall(r"\d+", segment)
        if len(nums) >= 2:
            start, end = int(nums[0]), int(nums[1])
            if 0 <= start < end <= len(source_text):
                spans.append((start, end))
    if not spans:
        return None
    source_segments = [source_text[start:end] for start, end in spans]
    joined = " ".join(source_segments)
    offset_match = _norm(joined) == _norm(mention)
    return {
        "annotation_id": annotation_id,
        "entity_type": entity_type,
        "mention": mention,
        "spans": spans,
        "spans_json": json.dumps([[a, b] for a, b in spans]),
        "start": min(a for a, _ in spans),
        "end": max(b for _, b in spans),
        "is_discontinuous": len(spans) > 1,
        "offset_text": joined,
        "offset_match": bool(offset_match),
    }


def _locate_cadec_root(path: str | Path) -> Path:
    root = Path(path)
    if (root / "cadec").exists():
        root = root / "cadec"
    required = [root / "text", root / "original", root / "meddra"]
    if not all(item.exists() for item in required):
        raise FileNotFoundError("Expected CADEC text/, original/, and meddra/ directories")
    return root


def parse_cadec(
    cadec_root: str | Path,
    output_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse CADEC MedDRA normalisations while preserving exact BRAT spans.

    Discontinuous annotations are stored as ``spans_json`` rather than collapsed into a
    single evidence interval. ``start``/``end`` remain the outer bounds for convenience,
    but downstream explainability should prefer ``spans_json`` when present.
    """
    root = _locate_cadec_root(cadec_root)
    text_dir, original_dir, meddra_dir = root / "text", root / "original", root / "meddra"

    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "documents_seen": 0,
        "documents_with_text": 0,
        "normalizations_seen": 0,
        "normalizations_matched": 0,
        "missing_target": 0,
        "invalid_textbound": 0,
        "discontinuous_annotations": 0,
        "offset_match_count": 0,
    }

    for meddra_path in sorted(meddra_dir.glob("*.ann")):
        stats["documents_seen"] += 1
        stem = meddra_path.stem
        text_path = text_dir / f"{stem}.txt"
        if not text_path.exists():
            continue
        stats["documents_with_text"] += 1
        text = text_path.read_text(encoding="utf-8", errors="replace")
        original_path = original_dir / f"{stem}.ann"
        annotation_lines: list[str] = []
        if original_path.exists():
            annotation_lines.extend(original_path.read_text(encoding="utf-8", errors="replace").splitlines())
        meddra_lines = meddra_path.read_text(encoding="utf-8", errors="replace").splitlines()
        annotation_lines.extend(meddra_lines)

        textbounds: dict[str, dict[str, Any]] = {}
        for line in annotation_lines:
            if not line.startswith("T"):
                continue
            parsed = _parse_textbound(line, text)
            if parsed is None:
                stats["invalid_textbound"] += 1
                continue
            textbounds[str(parsed["annotation_id"])] = parsed

        for line in meddra_lines:
            target_match = _TARGET_RE.search(line)
            code_match = _CODE_RE.search(line)
            if not (target_match and code_match):
                continue
            stats["normalizations_seen"] += 1
            target_id = target_match.group(1)
            bound = textbounds.get(target_id)
            if bound is None:
                stats["missing_target"] += 1
                continue
            stats["normalizations_matched"] += 1
            stats["discontinuous_annotations"] += int(bool(bound["is_discontinuous"]))
            stats["offset_match_count"] += int(bool(bound["offset_match"]))
            normalization_id = line.split("\t", 1)[0] if "\t" in line else ""
            rows.append({
                "record_id": stem,
                "annotation_id": target_id,
                "normalization_id": normalization_id,
                "text": text,
                "mention": bound["mention"],
                "entity_type": bound["entity_type"],
                "start": bound["start"],
                "end": bound["end"],
                "spans_json": bound["spans_json"],
                "is_discontinuous": bool(bound["is_discontinuous"]),
                "offset_text": bound["offset_text"],
                "offset_match": bool(bound["offset_match"]),
                "gold_code": code_match.group(1),
                "gold_term": "",
                "source_dataset": "CADEC",
            })

    if not rows:
        raise ValueError("No CADEC MedDRA normalisations parsed")
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(["record_id", "annotation_id", "gold_code"]).reset_index(drop=True)
    stats["duplicate_rows_removed"] = int(before - len(df))
    stats.update({
        "rows": int(len(df)),
        "unique_documents": int(df["record_id"].nunique()),
        "unique_codes": int(df["gold_code"].nunique()),
        "match_rate": stats["normalizations_matched"] / stats["normalizations_seen"] if stats["normalizations_seen"] else 0.0,
        "offset_match_rate": float(df["offset_match"].mean()) if len(df) else 0.0,
        "source_fingerprint_sha256": sha256("".join(sorted(df["record_id"].astype(str).unique())).encode()).hexdigest(),
    })

    if output_csv is not None:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        output.with_suffix(".parse_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return df, stats


def audit_cadec_records(
    records: pd.DataFrame,
    terminology: pd.DataFrame | None = None,
    *,
    minimum_offset_match_rate: float = 0.99,
    minimum_terminology_coverage: float = 0.95,
) -> dict[str, Any]:
    required = {"record_id", "text", "mention", "gold_code"}
    if not required.issubset(records.columns):
        raise ValueError(f"CADEC records require {sorted(required)}")
    df = records.copy().fillna("")
    duplicate_count = int(df.duplicated(["record_id", "annotation_id", "gold_code"]).sum()) if "annotation_id" in df else int(df.duplicated(["record_id", "mention", "gold_code"]).sum())
    offset_match_rate = float(df["offset_match"].astype(bool).mean()) if "offset_match" in df and len(df) else None
    multi_code = 0
    if "annotation_id" in df:
        multi_code = int((df.groupby(["record_id", "annotation_id"])["gold_code"].nunique() > 1).sum())

    terminology_coverage = None
    missing_codes: list[str] = []
    if terminology is not None and not terminology.empty:
        known = set(terminology["code"].astype(str))
        gold = set(df["gold_code"].astype(str))
        missing_codes = sorted(gold - known)
        terminology_coverage = 1.0 - (len(missing_codes) / len(gold) if gold else 0.0)

    hard_failures: list[str] = []
    warnings: list[str] = []
    if (df["gold_code"].astype(str).str.strip() == "").any():
        hard_failures.append("blank_gold_code")
    if duplicate_count:
        warnings.append("duplicate_rows_present")
    if offset_match_rate is not None and offset_match_rate < minimum_offset_match_rate:
        hard_failures.append("offset_match_rate_below_threshold")
    if terminology_coverage is not None and terminology_coverage < minimum_terminology_coverage:
        hard_failures.append("terminology_coverage_below_threshold")
    if multi_code:
        warnings.append("same_annotation_has_multiple_codes")

    return {
        "dataset": "CADEC",
        "rows": int(len(df)),
        "unique_documents": int(df["record_id"].nunique()),
        "unique_codes": int(df["gold_code"].nunique()),
        "duplicate_rows": duplicate_count,
        "offset_match_rate": offset_match_rate,
        "discontinuous_rows": int(df["is_discontinuous"].astype(bool).sum()) if "is_discontinuous" in df else None,
        "multi_code_annotations": multi_code,
        "terminology_coverage": terminology_coverage,
        "missing_terminology_codes": missing_codes[:100],
        "minimum_offset_match_rate": float(minimum_offset_match_rate),
        "minimum_terminology_coverage": float(minimum_terminology_coverage),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "ready_for_benchmark": not hard_failures,
    }


def make_manual_review_sample(records: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create an audit-first review sample rather than a purely random sample."""
    df = records.copy().fillna("")
    priority = pd.Series(False, index=df.index)
    if "offset_match" in df:
        priority |= ~df["offset_match"].astype(bool)
    if "is_discontinuous" in df:
        priority |= df["is_discontinuous"].astype(bool)
    selected = df[priority].copy()
    remaining = max(0, int(n) - len(selected))
    if remaining:
        pool = df[~df.index.isin(selected.index)]
        if len(pool) > remaining:
            pool = pool.sample(n=remaining, random_state=seed)
        selected = pd.concat([selected, pool], ignore_index=True)
    selected = selected.head(int(n)).copy()
    for column in ["review_span_correct", "review_code_link_correct", "review_comments"]:
        selected[column] = ""
    return selected


def write_cadec_audit_artifacts(
    records: pd.DataFrame,
    output_dir: str | Path,
    terminology: pd.DataFrame | None = None,
    *,
    sample_size: int = 50,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_cadec_records(records, terminology)
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    make_manual_review_sample(records, n=sample_size).to_csv(output / "manual_parser_review_sample.csv", index=False)
    return audit
