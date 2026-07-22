#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value):
    return "N/A" if value is None else f"{100*float(value):.1f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--publish-dir", required=True)
    p.add_argument("--max-examples", type=int, default=8)
    a = p.parse_args()
    run = Path(a.run_dir)
    pub = Path(a.publish_dir)
    pub.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "real_deepseek_results.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (run / "deepseek_real_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    (pub / "real_deepseek_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    examples = cases[: a.max_examples]
    (pub / "example_cases.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    b, d = summary["baseline"], summary["deepseek"]
    lines = [
        "# MedCode v0.1.1 — real MedNorm + DeepSeek evaluation",
        "",
        "> This is a real-data **closed-code** evaluation using public MedNorm-derived phrases and TRAIN-derived candidate aliases. It is not a full licensed-MedDRA open-set benchmark.",
        "",
        "## Observed results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Real baseline test sample | {b['n_baseline_test_sample']} cases |",
        f"| Baseline Accuracy@1 | {pct(b['baseline_accuracy_at_1'])} |",
        f"| Baseline Accuracy@3 | {pct(b['baseline_accuracy_at_3'])} |",
        f"| Baseline / candidate Recall@5 | {pct(b['candidate_recall_at_5'])} |",
        f"| Gold code seen in TRAIN rate | {pct(b['gold_seen_in_train_rate'])} |",
        f"| DeepSeek paired real-data subset | {d['n_deepseek_real_cases']} cases |",
        f"| Paired baseline Accuracy@1 | {pct(d['paired_baseline_accuracy_at_1'])} |",
        f"| DeepSeek reranked Accuracy@1 | {pct(d['deepseek_reranked_accuracy_at_1'])} |",
        f"| DeepSeek Accuracy@1 delta | {pct(d['deepseek_accuracy_delta'])} |",
        f"| Fixed candidate Recall@5 | {pct(d['fixed_candidate_recall_at_5'])} |",
        f"| Valid DeepSeek API response rate | {pct(d['deepseek_api_success_rate'])} |",
        "",
        "`Recall@5` is also reported as an **oracle Top-K human-choice upper bound**: it assumes a human always chooses the gold code whenever present and is not observed human-assisted accuracy.",
        "",
        "## Routing",
        "",
        "```json",
        json.dumps(b.get("route_counts", {}), indent=2),
        "```",
        "",
        "## Candidate rationale contract",
        "",
        "Every displayed option is tied to the real benchmark phrase. A valid DeepSeek response must preserve the exact fixed candidate code set and provide a rationale for every candidate citing at least one exact approved real-data phrase. Missing/invalid LLM output falls back to deterministic grounded rationale.",
        "",
        "## Example real-data cases",
        "",
    ]
    for case in examples:
        lines.extend([
            f"### {case['record_id']}", "",
            f"**Real phrase:** `{case['real_phrase']}`  ",
            f"**Gold:** `{case['gold_code']}` · **Baseline top1:** `{case['baseline_top1']}` · **DeepSeek top1:** `{case['deepseek_top1']}` · **Route:** `{case['route']}`", "",
        ])
        for option in case["candidate_options"]:
            lines.extend([
                f"- **#{option['deepseek_rank']} `{option['code']}` — {option['term']}**",
                f"  - Evidence: `{'; '.join(option['real_evidence_quotes'])}`",
                f"  - Rationale ({option['rationale_source']}): {option['deepseek_rationale']}",
            ])
        lines.append("")
    lines.extend([
        "## Data/source note", "",
        "MedNorm official source/licence authority: DOI `10.17632/b9x7xxb9sz.1`, CC BY-NC 3.0. The Hugging Face copy is used only as a transport convenience and does not override the official licence.",
    ])
    (pub / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
