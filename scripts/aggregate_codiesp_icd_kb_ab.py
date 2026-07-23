#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def codes(x: str) -> set[str]:
    return {v for v in str(x or "").split("|") if v and v != "nan"}


def aggregate(df: pd.DataFrame) -> dict:
    tp = fp = fn = exact = 0
    missed = []
    for _, r in df.iterrows():
        g, p = codes(r["gold"]), codes(r["pred"])
        tp += len(g & p); fp += len(p - g); fn += len(g - p); exact += int(g == p)
        missed.append(len(g - p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_documents": int(len(df)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_match": exact / len(df) if len(df) else 0.0,
        "mean_missed_codes_per_case": sum(missed) / len(missed) if missed else 0.0,
        "median_missed_codes_per_case": float(pd.Series(missed).median()) if missed else 0.0,
        "mean_gold_codes_per_case": float(df["gold"].map(lambda x: len(codes(x))).mean()) if len(df) else 0.0,
        "mean_predicted_codes_per_case": float(df["pred"].map(lambda x: len(codes(x))).mean()) if len(df) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    inp = Path(args.input_dir); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    all_frames = []
    manifests = []
    for p in inp.rglob("predictions.csv"):
        if p.stat().st_size:
            df = pd.read_csv(p, dtype=str).fillna("")
            all_frames.append(df)
    for p in inp.rglob("run_manifest.json"):
        manifests.append(json.loads(p.read_text(encoding="utf-8")))
    if not all_frames:
        raise RuntimeError("No predictions found")
    merged = pd.concat(all_frames, ignore_index=True)
    merged = merged.sort_values(["mode", "article_id"]).drop_duplicates(["mode", "article_id"], keep="last")
    merged.to_csv(out / "all_predictions.csv", index=False)

    modes = {}
    for mode in ["direct", "rag"]:
        x = merged[merged["mode"] == mode].copy()
        modes[mode] = aggregate(x)
        if mode == "rag" and len(x):
            modes[mode]["mean_retrieval_gold_recall"] = float(pd.to_numeric(x["retrieval_gold_recall"], errors="coerce").fillna(0).mean())
        x.to_csv(out / f"{mode}_predictions.csv", index=False)

    d = merged[merged["mode"] == "direct"].set_index("article_id")
    r = merged[merged["mode"] == "rag"].set_index("article_id")
    common = sorted(set(d.index) & set(r.index))
    rows = []
    for aid in common:
        dg, dp, rp = codes(d.loc[aid, "gold"]), codes(d.loc[aid, "pred"]), codes(r.loc[aid, "pred"])
        def f1(g, p):
            tp=len(g&p); fp=len(p-g); fn=len(g-p); pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
            return 2*pr*rc/(pr+rc) if pr+rc else 0
        rows.append({
            "article_id": aid,
            "direct_f1": f1(dg, dp),
            "rag_f1": f1(dg, rp),
            "delta_f1": f1(dg, rp)-f1(dg, dp),
            "direct_missed": len(dg-dp),
            "rag_missed": len(dg-rp),
            "delta_missed": len(dg-rp)-len(dg-dp),
        })
    cmp = pd.DataFrame(rows)
    cmp.to_csv(out / "paired_case_comparison.csv", index=False)
    summary = {
        "schema_version": "codiesp-icd-kb-ab-summary-v0.1",
        "strictly_paired_case_ids": common,
        "n_paired": len(common),
        "metrics": modes,
        "paired": {
            "rag_better_f1_cases": int((cmp["delta_f1"] > 0).sum()) if len(cmp) else 0,
            "direct_better_f1_cases": int((cmp["delta_f1"] < 0).sum()) if len(cmp) else 0,
            "tie_f1_cases": int((cmp["delta_f1"] == 0).sum()) if len(cmp) else 0,
            "rag_reduced_missed_codes_cases": int((cmp["delta_missed"] < 0).sum()) if len(cmp) else 0,
            "rag_increased_missed_codes_cases": int((cmp["delta_missed"] > 0).sum()) if len(cmp) else 0,
        },
        "manifests": manifests,
        "scientific_note": "Direct and RAG modes use the same fixed CodiEsp test cases, same DeepSeek V4-Pro model, thinking enabled with max reasoning effort, and same output/evaluation schema. The RAG arm differs only by dynamic retrieval of FY2018 ICD-10-CM structured knowledge. Retrieved codes are not a whitelist.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
