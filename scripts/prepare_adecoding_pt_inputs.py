from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def parse_list(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    return [str(item).strip() for item in parsed if str(item).strip()]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def error_type(predicted: list[str], target: list[str]) -> str:
    p, t = set(predicted), set(target)
    if p == t:
        return "EXACT"
    if not p & t:
        return "DISJOINT"
    if p > t:
        return "OVER_PREDICTION"
    if t > p:
        return "UNDER_PREDICTION"
    return "MIXED_PARTIAL_OVERLAP"


def quantile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as archive:
        member = next(name for name in archive.namelist() if name.endswith("sample_errors.csv"))
        payload = archive.read(member)
    source_sha = hashlib.sha256(payload).hexdigest()
    source_rows = []
    groups: dict[str, list[dict]] = defaultdict(list)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    for ordinal, raw in enumerate(reader, start=1):
        text = (raw.get("Text") or "").strip()
        predicted = parse_list(raw.get("Predicted") or "")
        target = parse_list(raw.get("Target") or "")
        normalized = normalize_text(text)
        row = {
            "source_ordinal": ordinal,
            "source_row": (raw.get("") or str(ordinal)).strip(),
            "input_text": text,
            "normalized_text": normalized,
            "predicted": predicted,
            "predicted_probs": parse_list(raw.get("Predicted Probs") or ""),
            "target": target,
        }
        source_rows.append(row)
        groups[normalized].append(row)

    lengths = [len(row["input_text"]) for row in source_rows]
    q1, q2 = quantile(lengths, 1 / 3), quantile(lengths, 2 / 3)
    group_meta = {}
    for normalized, rows in groups.items():
        target_sets = {tuple(row["target"]) for row in rows}
        group_meta[normalized] = {
            "duplicate_count": len(rows),
            "target_conflict": len(target_sets) > 1,
        }

    output_rows = []
    for ordinal, row in enumerate(source_rows, start=1):
        length = len(row["input_text"])
        length_bucket = "SHORT" if length <= q1 else "MEDIUM" if length <= q2 else "LONG"
        meta = group_meta[row["normalized_text"]]
        output_rows.append({
            "sample_id": f"ADE-PT-ALL-{ordinal:05d}",
            "source_row": row["source_row"],
            "source_ordinal": row["source_ordinal"],
            "input_text": row["input_text"],
            "input_sha256": hashlib.sha256(row["input_text"].encode("utf-8")).hexdigest(),
            "normalized_text_sha256": hashlib.sha256(row["normalized_text"].encode("utf-8")).hexdigest(),
            "legacy_target_pt_terms_json": json.dumps(row["target"], ensure_ascii=False, separators=(",", ":")),
            "legacy_target_pt_codes_json": "[]",
            "legacy_predicted_pt_terms_json": json.dumps(row["predicted"], ensure_ascii=False, separators=(",", ":")),
            "legacy_predicted_probabilities_json": json.dumps(row["predicted_probs"], ensure_ascii=False, separators=(",", ":")),
            "target_cardinality": len(row["target"]),
            "target_cardinality_bucket": "SINGLE" if len(row["target"]) == 1 else "DOUBLE" if len(row["target"]) == 2 else "THREE_PLUS",
            "text_length_chars": length,
            "text_length_bucket": length_bucket,
            "legacy_error_type": error_type(row["predicted"], row["target"]),
            "source_duplicate_count": meta["duplicate_count"],
            "duplicate_target_conflict": meta["target_conflict"],
            "input_status": "NONEMPTY" if row["input_text"] else "EMPTY_SOURCE_TEXT",
            "data_classification": "restricted",
            "legacy_meddra_version": "UNKNOWN",
            "reference_status": "LEGACY_TEST_TARGET_NOT_INDEPENDENT_CLINICAL_GOLD",
            "selection_strategy": "ALL_SOURCE_ROWS_PRESERVE_ORDER_WITH_DUPLICATES_V1",
            "source_member_sha256": source_sha,
        })

    output_path = output_dir / "adecoding_pt_all_inputs.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    output_path.chmod(0o600)
    manifest = {
        "schema_version": "1.0",
        "source_member": member,
        "source_member_sha256": source_sha,
        "source_row_count": len(source_rows),
        "output_row_count": len(output_rows),
        "normalized_unique_text_count": len(groups),
        "duplicate_row_count": len(source_rows) - len(groups),
        "conflicting_duplicate_group_count": sum(meta["target_conflict"] for meta in group_meta.values()),
        "empty_text_count": sum(not row["input_text"] for row in source_rows),
        "target_cardinality_counts": dict(sorted(Counter(row["target_cardinality"] for row in output_rows).items())),
        "data_classification": "restricted",
        "meddra_version": "UNKNOWN",
        "reference_semantics": "legacy test targets; not independent clinical ground truth",
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    manifest_path = output_dir / "all_inputs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.chmod(0o600)
    print(json.dumps(manifest, separators=(",", ":")))


if __name__ == "__main__":
    main()
