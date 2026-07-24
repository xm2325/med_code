#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

import run_codiesp_direct_deepseek_thinking as base


def run_prompt(name: str, ids: list[str], texts: dict[str, str], gold: dict[str, set[str]], api_key: str, out_csv: Path):
    pred: dict[str, set[str]] = {}
    rows = []
    failures = []
    template = base.PROMPTS[name]
    for index, doc_id in enumerate(ids, start=1):
        try:
            # Replace only the explicit case placeholder; do not interpret JSON-schema braces in the prompt.
            prompt = template.replace("{case}", texts[doc_id])
            obj = base.call_deepseek(prompt, api_key)
            codes = {base.normalize_code(c) for c in obj.get("codes", []) if base.normalize_code(c)}
            items = obj.get("items", []) if isinstance(obj.get("items", []), list) else []
            pred[doc_id] = codes
            rows.append({
                "article_id": doc_id,
                "prompt_variant": name,
                "gold": "|".join(sorted(gold.get(doc_id, set()))),
                "pred": "|".join(sorted(codes)),
                "items_json": json.dumps(items, ensure_ascii=False),
            })
        except Exception as exc:  # noqa: BLE001
            pred[doc_id] = set()
            failures.append({"article_id": doc_id, "prompt_variant": name, "error": str(exc)})
        if index % 10 == 0:
            print(f"{name}: {index}/{len(ids)}", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(failures).to_csv(out_csv.with_name(out_csv.stem + "_failures.csv"), index=False)
    m = base.metrics(gold, pred, ids)
    m["n_failed_requests"] = len(failures)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dev-selection-n", type=int, default=40)
    ap.add_argument("--test-n", type=int, default=250)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    dev_text, test_text, dev_gold, test_gold = base.load_codiesp()
    dev_ids = sorted(set(dev_text) & set(dev_gold))[:args.dev_selection_n]
    test_ids = sorted(set(test_text) & set(test_gold))[:args.test_n]

    dev_results = {}
    for name in base.PROMPTS:
        dev_results[name] = run_prompt(name, dev_ids, dev_text, dev_gold, api_key, out / f"dev_{name}_predictions.csv")

    valid = [n for n in base.PROMPTS if dev_results[n]["n_failed_requests"] == 0]
    if not valid:
        raise RuntimeError(f"All prompt variants failed API execution: {dev_results}")
    winner = sorted(valid, key=lambda n: (dev_results[n]["micro_f1"], dev_results[n]["micro_recall"]), reverse=True)[0]
    test_metrics = run_prompt(winner, test_ids, test_text, test_gold, api_key, out / "test_direct_thinking_predictions.csv")
    if test_metrics["n_failed_requests"] != 0:
        raise RuntimeError(f"Test execution had failed requests: {test_metrics['n_failed_requests']}")

    summary = {
        "schema_version": "codiesp-direct-deepseek-thinking-v0.2",
        "model": base.MODEL,
        "thinking": {"type": "enabled", "reasoning_effort": "max"},
        "no_retrieval_or_candidate_list": True,
        "dev_prompt_selection": {
            "n_documents": len(dev_ids),
            "selection_rule": "highest micro-F1 on deterministic first dev IDs; micro-recall tie-break",
            "results": dev_results,
            "selected_prompt": winner,
        },
        "test": {"n_documents": len(test_ids), "selected_prompt": winner, "metrics": test_metrics},
        "prompts": base.PROMPTS,
        "scientific_note": "Prompt selection used only development cases. Test metrics use the frozen selected prompt. No TF-IDF, similar-case retrieval, candidate code list, or external code-description lookup was provided to the model.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
