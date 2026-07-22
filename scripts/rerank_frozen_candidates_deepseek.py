#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.llm_rerank import DeepSeekCandidateReranker


def _attach_queries(records: pd.DataFrame, predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    source = records[records["split"].astype(str) == str(split)].copy()
    source["_occurrence"] = source.groupby("record_id").cumcount()
    pred = predictions.copy()
    pred["_occurrence"] = pred.groupby("record_id").cumcount()
    cols = ["record_id", "_occurrence", "text"]
    if "mention" in source:
        cols.append("mention")
    merged = pred.merge(source[cols], on=["record_id", "_occurrence"], how="left", validate="one_to_one")
    if merged["text"].isna().any():
        raise ValueError("Could not align every prediction to source text")
    return merged.drop(columns=["_occurrence"])


def main() -> None:
    p = argparse.ArgumentParser(description="Rerank a frozen candidate set with a fixed DeepSeek prompt")
    p.add_argument("--records", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--model", default="deepseek-v4-pro")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--max-records", type=int)
    p.add_argument("--allow-external-llm", action="store_true")
    p.add_argument("--data-classification", choices=["public", "synthetic", "restricted", "private"], default="restricted")
    args = p.parse_args()

    records = pd.read_csv(args.records, dtype=str, keep_default_na=False)
    predictions = pd.read_csv(args.predictions, dtype=str, keep_default_na=False)
    aligned = _attach_queries(records, predictions, args.split)
    if args.max_records is not None:
        aligned = aligned.head(max(0, int(args.max_records))).copy()

    client = DeepSeekCandidateReranker(model=args.model)
    rows = []
    n_accepted = 0
    for _, row in aligned.iterrows():
        candidates = json.loads(row.get("candidates_json", "[]"))[: max(1, int(args.top_n))]
        query = str(row.get("mention", "") or row.get("text", ""))
        result = client.rerank(
            query,
            candidates,
            allow_external_llm=args.allow_external_llm,
            data_classification=args.data_classification,
        )
        reordered = list(candidates)
        if result["accepted"]:
            order = [str(code) for code in result["payload"]["ranked_codes"]]
            by_code = {str(item["code"]): item for item in candidates}
            reordered = [by_code[code] for code in order]
            n_accepted += 1
        predicted_code = str(reordered[0]["code"]) if reordered else str(row.get("predicted_code", ""))
        rows.append({
            "record_id": str(row.get("record_id", "")),
            "gold_code": str(row.get("gold_code", "")),
            "original_predicted_code": str(row.get("predicted_code", "")),
            "reranked_predicted_code": predicted_code,
            "correct": int(predicted_code == str(row.get("gold_code", ""))) if str(row.get("gold_code", "")) else "",
            "rerank_accepted": bool(result["accepted"]),
            "rerank_validation_errors_json": json.dumps(result.get("validation_errors", [])),
            "rerank_payload_json": json.dumps(result.get("payload"), ensure_ascii=False),
            "reranked_candidates_json": json.dumps(reordered, ensure_ascii=False),
        })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "reranked_predictions.csv", index=False)
    metrics = {
        "version": "0.0.13",
        "split": args.split,
        "prompt_version": "deepseek_candidate_rerank_v1",
        "model": args.model,
        "n": len(frame),
        "n_reranks_accepted": n_accepted,
        "rerank_acceptance_rate": n_accepted / len(frame) if len(frame) else 0.0,
        "accuracy_at_1": float(pd.to_numeric(frame["correct"], errors="coerce").mean()) if len(frame) and frame["correct"].astype(str).str.len().gt(0).any() else None,
        "selection_warning": "Fix the model/prompt policy on development/validation data before evaluating TEST. This script does not tune on TEST.",
        "candidate_set_locked": True,
        "data_classification": args.data_classification,
    }
    (output / "rerank_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
