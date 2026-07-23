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
    cases_path = run / "deepseek_real_cases.jsonl"
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if cases_path.exists() else []

    (pub / "real_deepseek_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    examples = cases[: a.max_examples]
    (pub / "example_cases.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    b, d = summary["baseline"], summary["deepseek"]
    deepseek_completed = bool(d.get("n_valid_deepseek_responses", 0))

    lines = [
        "# MedCode v0.1.1 — real MedNorm evaluation",
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
        f"| Paired fixed-seed subset | {d.get('n_paired_subset_cases', 0)} cases |",
        f"| Paired baseline Accuracy@1 | {pct(d.get('paired_baseline_accuracy_at_1'))} |",
        f"| DeepSeek API calls attempted | {d.get('n_deepseek_real_cases', 0)} |",
        f"| Valid DeepSeek responses | {d.get('n_valid_deepseek_responses', 0)} |",
        f"| DeepSeek accepted-only Accuracy@1 | {pct(d.get('deepseek_reranked_accuracy_at_1_accepted_only'))} |",
        f"| DeepSeek accepted-only Accuracy@1 delta | {pct(d.get('deepseek_accuracy_delta_accepted_only'))} |",
        f"| Fixed candidate Recall@5 | {pct(d.get('fixed_candidate_recall_at_5'))} |",
        f"| Valid DeepSeek API response rate | {pct(d.get('deepseek_api_success_rate'))} |",
        "",
        "`Recall@5` is also an **oracle Top-K human-choice upper bound**: it assumes a human always chooses the gold code whenever present and is not observed human-assisted accuracy.",
        "",
    ]

    if not deepseek_completed:
        lines.extend([
            "## DeepSeek execution status",
            "",
            "No valid DeepSeek response is included in this run. The real-data baseline and fixed Top-K subset remain valid and reproducible; candidate explanations shown below use deterministic grounded rationale until the repository Actions secret is available.",
            "",
        ])

    lines.extend([
        "## Routing",
        "",
        "```json",
        json.dumps(b.get("route_counts", {}), indent=2),
        "```",
        "",
        "## Candidate rationale contract",
        "",
        "Every displayed option is tied to the real benchmark phrase. DeepSeek output, when available, must preserve the exact fixed candidate code set and provide one rationale per candidate citing exact approved real-data evidence. When DeepSeek is unavailable, every option still carries deterministic grounded evidence/rationale and is explicitly labelled as such.",
        "",
        "## Example real-data cases",
        "",
    ])

    for case in examples:
        deepseek_top1 = case.get("deepseek_top1") or "not run / no valid response"
        lines.extend([
            f"### {case['record_id']}",
            "",
            f"**Real phrase:** `{case['real_phrase']}`  ",
            f"**Gold:** `{case['gold_code']}` · **Baseline top1:** `{case['baseline_top1']}` · **DeepSeek top1:** `{deepseek_top1}` · **Route:** `{case['route']}`",
            "",
        ])
        for option in case["candidate_options"]:
            displayed_rank = option.get("deepseek_rank") or option.get("baseline_rank")
            rationale = option.get("display_rationale") or option.get("deepseek_rationale") or ""
            lines.extend([
                f"- **#{displayed_rank} `{option['code']}` — {option['term']}**",
                f"  - Evidence: `{'; '.join(option['real_evidence_quotes'])}`",
                f"  - Rationale ({option['rationale_source']}): {rationale}",
            ])
        lines.append("")

    lines.extend([
        "## Data/source note",
        "",
        "MedNorm official source/licence authority: DOI `10.17632/b9x7xxb9sz.1`, CC BY-NC 3.0. The Hugging Face copy is used only as a transport convenience and does not override the official licence.",
    ])
    (pub / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
