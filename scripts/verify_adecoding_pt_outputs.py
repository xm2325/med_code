from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


METHODS = ("prompt_only", "extract_lexical", "hybrid_rag")
EXPECTED_SAMPLES = 5921


def parse_json(value: str):
    return json.loads(value)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_all_outputs.py OUTPUT_DIRECTORY")
    root = Path(sys.argv[1]).resolve()
    long_path = root / "adecoding_pt_all_method_results_long.csv"
    wide_path = root / "adecoding_pt_all_comparison_wide.csv"
    summary_path = root / "adecoding_pt_all_summary.csv"
    manifest_path = root / "run_manifest.json"
    input_path = root / "adecoding_pt_all_inputs.csv"
    vocabulary_path = root / "legacy_pt_vocabulary.csv"

    for path in (long_path, wide_path, summary_path, manifest_path, input_path, vocabulary_path):
        require(path.is_file(), f"missing required file: {path.name}")

    input_rows = read_csv(input_path)
    long_rows = read_csv(long_path)
    wide_rows = read_csv(wide_path)
    summary_rows = read_csv(summary_path)
    vocabulary_rows = read_csv(vocabulary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    require(len(input_rows) == EXPECTED_SAMPLES, f"input row count: {len(input_rows)}")
    require(len(wide_rows) == EXPECTED_SAMPLES, f"wide row count: {len(wide_rows)}")
    require(len(long_rows) == EXPECTED_SAMPLES * len(METHODS), f"long row count: {len(long_rows)}")
    require(len(summary_rows) == 4, f"summary row count: {len(summary_rows)}")
    require(len({row["sample_id"] for row in input_rows}) == EXPECTED_SAMPLES, "input sample IDs are not unique")
    require([row["sample_id"] for row in wide_rows] == [row["sample_id"] for row in input_rows], "wide source order differs from input")
    require(manifest["input_count"] == EXPECTED_SAMPLES, "manifest input_count mismatch")
    require(manifest["method_result_count"] == EXPECTED_SAMPLES * len(METHODS), "manifest method_result_count mismatch")
    require(manifest["meddra_version"] == "UNKNOWN", "unexpected MedDRA version claim")
    require(manifest["pt_code_status"] == "UNRESOLVED_NO_VERSIONED_PT_ASC", "unexpected PT code status")
    require(manifest["clinical_accuracy_claimed"] is False, "manifest must not claim clinical accuracy")

    expected_hashes = {
        long_path.name: sha256_file(long_path),
        wide_path.name: sha256_file(wide_path),
        summary_path.name: sha256_file(summary_path),
    }
    require(manifest["outputs"] == expected_hashes, "output hash mismatch")
    require(manifest["input_sha256"] == sha256_file(input_path), "input hash mismatch")
    require(manifest["vocabulary_sha256"] == sha256_file(vocabulary_path), "vocabulary hash mismatch")

    allowed_ids = {row["legacy_term_id"] for row in vocabulary_rows}
    allowed_terms = {row["pt_term"] for row in vocabulary_rows}
    require(len(allowed_ids) == len(vocabulary_rows), "duplicate vocabulary IDs")
    require(all(not row.get("pt_code") for row in vocabulary_rows), "unexpected PT code in legacy vocabulary")

    input_by_id = {row["sample_id"]: row for row in input_rows}
    long_index: dict[tuple[str, str], dict[str, str]] = {}
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    route_counts: dict[str, Counter[str]] = defaultdict(Counter)
    validation_errors = Counter()
    offset_event_count = 0
    hybrid_event_count = 0
    duplicate_reuse_count = Counter()

    for row in long_rows:
        sample_id = row["sample_id"]
        method = row["method_id"]
        require(sample_id in input_by_id, f"unknown sample {sample_id}")
        require(method in METHODS, f"unknown method {method}")
        key = (sample_id, method)
        require(key not in long_index, f"duplicate long row {key}")
        long_index[key] = row
        source = input_by_id[sample_id]
        require(row["input_text"] == source["input_text"], f"input mismatch {key}")
        require(row["input_sha256"] == hashlib.sha256(row["input_text"].encode("utf-8")).hexdigest(), f"text hash mismatch {key}")
        require(parse_json(row["selected_pt_codes_json"]) == [], f"invented PT code {key}")
        require(row["pt_code_status"] == "UNRESOLVED_NO_VERSIONED_PT_ASC", f"PT code status mismatch {key}")
        require(row["reference_meddra_version"] == "UNKNOWN", f"reference version mismatch {key}")
        require(as_bool(row["causality_not_assessed"]), f"causality flag false {key}")
        require(as_bool(row["severity_not_assessed"]), f"severity flag false {key}")
        require(not as_bool(row["clinical_accuracy_claimed"]), f"clinical accuracy claimed {key}")
        require(row["execution_status"] in {"LIVE", "VALIDATION_FAILED", "ERROR"}, f"bad status {key}")
        require(row["route"] in {"AUTO_CANDIDATE", "TOP_K_HUMAN_CHOICE", "FULL_EXPERT_REVIEW"}, f"bad route {key}")
        selected_terms = parse_json(row["selected_pt_terms_json"])
        require(isinstance(selected_terms, list), f"selected terms not list {key}")
        require(all(term in allowed_terms for term in selected_terms), f"term outside vocabulary {key}")
        errors = parse_json(row["validation_errors_json"])
        require(isinstance(errors, list), f"validation errors not list {key}")
        for error in errors:
            validation_errors[(method, str(error))] += 1
        status_counts[method][row["execution_status"]] += 1
        route_counts[method][row["route"]] += 1
        if as_bool(row["inference_reused_from_duplicate"]):
            duplicate_reuse_count[method] += 1

        events = parse_json(row["events_json"])
        require(isinstance(events, list), f"events not list {key}")
        for event in events:
            quote = event["quote"]
            start, end = int(event["start"]), int(event["end"])
            require(event["offset_source"] == "SERVER_UNIQUE_EXACT_QUOTE_MATCH", f"offset source mismatch {key}")
            require(row["input_text"][start:end] == quote, f"verbatim offset mismatch {key}")
            require(row["input_text"].count(quote) == 1, f"quote is not unique {key}")
            offset_event_count += 1
            if method == "prompt_only":
                term_ids = event.get("term_ids", [])
                require(len(term_ids) == len(set(term_ids)), f"duplicate prompt term ID {key}")
                require(all(term_id in allowed_ids for term_id in term_ids), f"prompt term outside allowlist {key}")
                require([entry["legacy_term_id"] for entry in event.get("selected", [])] == term_ids, f"prompt selected mismatch {key}")
            elif method == "extract_lexical":
                candidates = event.get("candidates", [])
                selected = event.get("selected")
                if selected is not None:
                    require(bool(candidates), f"lexical selected without candidate {key}")
                    require(selected["legacy_term_id"] == candidates[0]["legacy_term_id"], f"lexical selected is not rank 1 {key}")
            else:
                retrieved = event.get("retrieved_candidates", [])
                reranked = event.get("reranked_candidates", [])
                selected = event.get("selected")
                if retrieved:
                    retrieved_ids = [entry["legacy_term_id"] for entry in retrieved]
                    reranked_ids = [entry["legacy_term_id"] for entry in reranked]
                    require(len(retrieved_ids) == len(set(retrieved_ids)), f"duplicate retrieved candidate {key}")
                    require(len(reranked_ids) == len(set(reranked_ids)), f"duplicate reranked candidate {key}")
                    require(event["candidate_set_sha256"] == sha256_json(retrieved_ids), f"candidate hash mismatch {key}")
                    if reranked_ids:
                        require(set(retrieved_ids) == set(reranked_ids), f"accepted frozen candidate set changed {key}")
                        require(selected is not None and selected["legacy_term_id"] == reranked_ids[0], f"RAG selected mismatch {key}")
                        hybrid_event_count += 1
                    else:
                        require(row["execution_status"] == "VALIDATION_FAILED", f"empty RAG rerank was not rejected {key}")
                        require("FROZEN_CANDIDATE_SET_CHANGED" in errors, f"missing frozen-set rejection evidence {key}")
                        require(selected is None, f"rejected RAG event retained a selection {key}")
                else:
                    require(selected is None and reranked == [], f"RAG empty candidate mismatch {key}")

    require(len(long_index) == EXPECTED_SAMPLES * len(METHODS), "long key cardinality mismatch")

    for wide in wide_rows:
        sample_id = wide["sample_id"]
        for method in METHODS:
            long = long_index[(sample_id, method)]
            for field in (
                "execution_status",
                "selected_pt_terms_json",
                "selected_pt_codes_json",
                "pt_code_status",
                "route",
                "review_reasons_json",
                "validation_errors_json",
                "legacy_target_exact_agreement",
                "legacy_target_precision",
                "legacy_target_recall",
                "legacy_target_f1",
                "latency_ms",
            ):
                require(str(wide[f"{method}__{field}"]) == str(long[field]), f"wide/long mismatch {sample_id}/{method}/{field}")
            require(str(wide[f"{method}__inference_representative_sample_id"]) == str(long["inference_representative_sample_id"]), f"reuse representative mismatch {sample_id}/{method}")

    summary_by_method = {row["method_id"]: row for row in summary_rows}
    require(set(summary_by_method) == {"legacy_clinicalbert_replay", *METHODS}, "summary method set mismatch")
    for method in METHODS:
        row = summary_by_method[method]
        subset = [entry for entry in long_rows if entry["method_id"] == method]
        require(int(row["sample_count"]) == EXPECTED_SAMPLES, f"summary sample count {method}")
        require(int(row["live_count"]) == status_counts[method]["LIVE"], f"summary live count {method}")
        require(int(row["validation_failed_count"]) == status_counts[method]["VALIDATION_FAILED"], f"summary validation count {method}")
        require(int(row["error_count"]) == status_counts[method]["ERROR"], f"summary error count {method}")
        exact_rate = statistics.mean(as_bool(entry["legacy_target_exact_agreement"]) for entry in subset)
        require(math.isclose(float(row["legacy_target_exact_agreement_rate"]), exact_rate, abs_tol=1e-6), f"summary agreement {method}")
        require(int(row["duplicate_reuse_count"]) == duplicate_reuse_count[method], f"summary duplicate reuse {method}")
        require(str(row["clinical_accuracy_claimed"]).lower() == "false", f"summary clinical claim {method}")

    require(not list(root.rglob("*.pdf")), "PDF found in private result directory")
    require(not list(root.rglob("*.zip")), "ZIP found in private result directory")
    report = {
        "verification": "passed",
        "sample_count": EXPECTED_SAMPLES,
        "method_result_count": len(long_rows),
        "offset_event_count": offset_event_count,
        "hybrid_frozen_candidate_event_count": hybrid_event_count,
        "status_counts": {method: dict(status_counts[method]) for method in METHODS},
        "route_counts": {method: dict(route_counts[method]) for method in METHODS},
        "validation_errors": {f"{method}|{error}": count for (method, error), count in validation_errors.items()},
        "duplicate_reuse_count": dict(duplicate_reuse_count),
        "output_sha256": expected_hashes,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
