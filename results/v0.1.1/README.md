# v0.1.1 real MedNorm + DeepSeek result status

The real-data evaluation has now completed successfully on public MedNorm-derived data with CADEC as the held-out TEST source.

Authoritative optimized run: `29987846810`  
Result commit: `f0c4ddd504e722675266dcee12ac0762e35b76ed`

## Core observed results

| Metric | Observed result |
|---|---:|
| Real TEST sample | 500 cases |
| Baseline Accuracy@1 | 57.0% |
| Baseline Accuracy@3 | 68.0% |
| Candidate Recall@5 | 71.8% |
| AUTO_CANDIDATE | 195 / 500 (39.0%) |
| AUTO_CANDIDATE Accuracy@1 | 95.38% |
| TOP_K_HUMAN_CHOICE | 305 / 500 (61.0%) |
| TOP_K_HUMAN_CHOICE Recall@5 | 55.08% |
| Fixed DeepSeek paired subset | 24 cases |
| Paired baseline Accuracy@1 | 41.67% |
| DeepSeek reranked Accuracy@1 | 54.17% |
| Paired change | +12.5 percentage points |
| Valid DeepSeek responses | 24 / 24 (100%) |
| Fixed candidate Recall@5 on paired subset | 70.83% |

The paired 24-case comparison changed 3 previously incorrect top-1 predictions to correct predictions and did not reverse any baseline-correct case in that fixed subset. This is a small paired experiment and should not be treated as definitive evidence of model superiority without a larger repeated evaluation.

## Explainability contract

Every displayed candidate option carries:

- the real benchmark phrase as verbatim source evidence;
- candidate code and terminology support;
- historical TRAIN provenance when available;
- a candidate-specific rationale;
- a clear rationale source (`deepseek_validated` or deterministic grounded fallback).

DeepSeek must preserve the fixed candidate code set. Invalid or ungrounded LLM output is rejected rather than silently accepted.

## Important interpretation boundaries

This is a **real-data closed-code diagnostic** using TRAIN-derived MedDRA code aliases. It is not a full licensed-MedDRA open-set benchmark. In the 500-case TEST sample, 4.6% of gold codes were unseen in TRAIN and their Accuracy@1 was 0% because those codes were structurally unavailable in the closed candidate space.

`Recall@5` is an oracle Top-K human-choice upper bound only under the assumption that a human always selects the gold code whenever it appears. It is **not observed human-assisted accuracy**.

See `real_deepseek_results.json`, `RESULTS.md`, `example_cases.json`, and `release/v0.1.1.json` for the machine-readable results and example rationale packages.
