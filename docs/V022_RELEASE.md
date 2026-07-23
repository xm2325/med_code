# MedCode v0.2.2 — real-data research milestone

v0.2.2 is the first MedCode milestone where the main public-data path has been tested as a connected application pipeline:

```text
real medical phrase
    ↓
alias-aware terminology + historical expert-memory candidate generation
    ↓
ranked MedDRA candidates
    ↓
DeepSeek fixed-candidate reranking where requested
    ↓
real phrase evidence + separate rationale for every displayed candidate
    ↓
target-domain calibrated routing
    ├── AUTO_CANDIDATE
    └── TOP_K_HUMAN_CHOICE / expert review
```

It is a research milestone, not a claim that BSRBR or another clinical cohort is ready for automatic coding.

## 1. Candidate generation: observed real-data improvement

Evaluation design:

- public MedNorm-derived records;
- CADEC-derived held-out target source;
- 3,000 model-training sample;
- 200 validation records for candidate-model selection;
- fixed 500-record diagnostic TEST sample;
- seed `20260723`;
- TRAIN-derived MedDRA candidate space only.

All candidate-model configuration selection was performed on VALIDATION before TEST scoring.

| Metric | v0.1.1 family | v0.2 alias-aware hybrid |
|---|---:|---:|
| Accuracy@1 | 57.0% | 67.0% |
| Recall@5 | 71.8% | 84.4% |
| Recall@20 | 75.4% | 88.6% |
| Recall@50 | 76.6% | 90.0% |
| Top-5 candidate failures | 141 | 78 |

The v0.2 retrieval configuration selected on validation was:

- historical-memory weight: `0.25`;
- word-TFIDF weight inside terminology retrieval: `0.15`;
- remaining terminology score from character n-gram retrieval.

The main change is that preferred terms **and all stored TRAIN aliases/synonyms** are now indexed as retrieval units before scores are aggregated to code level. The previous baseline stored many synonyms but did not use all of them directly in terminology retrieval.

## 2. DeepSeek paired reranking

The frozen v0.2 candidate generator was evaluated with `deepseek-v4-pro` on a fixed-seed 50-case real-data subset.

| Metric | Result |
|---|---:|
| Valid DeepSeek responses | 49 / 50 |
| Retrieval Accuracy@1 on paired subset | 56.0% |
| Pipeline Accuracy@1 with DeepSeek/fallback | 62.0% |
| Absolute change | +6.0 percentage points |
| Fixed candidate Recall@5 | 78.0% |
| Cases corrected by DeepSeek | 5 |
| Cases harmed by DeepSeek | 2 |
| Net additional correct cases | +3 |

The candidate code set is fixed before the LLM call. A valid DeepSeek response must preserve that exact code set and return one candidate-specific rationale per option with approved real source evidence. Invalid responses fall back to deterministic grounded rationale/ranking.

This 50-case paired result is supportive evidence, not a definitive superiority claim.

## 3. Why the first v0.2 AUTO policy was rejected

The first v0.2 end-to-end diagnostic selected a 95% target threshold on a small validation subset that had already been used for candidate-model selection.

It produced:

- AUTO coverage: 65.2%;
- held-out AUTO Accuracy@1: 84.0%.

That policy is **not release-eligible**.

v0.2.1 then separated model-selection validation from policy calibration, but the policy calibration data still came from a different source distribution than CADEC. Its 95% lower-bound policy reached only 91.1% Accuracy@1 on CADEC held-out TEST. That policy is also **not release-eligible**.

These failures are retained as diagnostic evidence rather than hidden.

## 4. v0.2.2 target-domain policy confirmation

v0.2.2 treats coding-policy calibration as a target-domain task.

The original 500 CADEC diagnostic TEST records were excluded entirely. From the remaining CADEC records:

- 1,000 fresh records were used for policy calibration;
- another 1,000 fresh records were used for confirmatory TEST;
- overlap between prior diagnostic, calibration and confirmatory partitions was exactly zero.

Candidate-model parameters remained frozen from the earlier non-CADEC validation-only model selection. CADEC labels affected the routing threshold only, not candidate-model fitting.

The prespecified operational target was 95% exact-code agreement. The policy selected the maximum calibration coverage whose one-sided binomial lower accuracy bound was at least 95%.

Calibration result:

- threshold: `0.5165572803781319`;
- AUTO coverage: 48.2%;
- empirical Accuracy@1: 96.68%;
- one-sided lower bound: 95.00%.

Locked fresh confirmatory result:

- AUTO coverage: **48.5%**;
- AUTO Accuracy@1: **96.91%**;
- prespecified 95% release gate: **PASS**.

This is the current public-data candidate AUTO policy result for v0.2.2.

## 5. Explanation output

Every displayed candidate keeps separate:

1. the original real phrase as source evidence;
2. terminology support and matched alias;
3. historical TRAIN provenance where available;
4. candidate-specific rationale;
5. DeepSeek or deterministic rationale source;
6. uncertainty/review route.

A plausible rationale does not make a code correct. Explanations support review and audit; they do not override held-out accuracy or routing policy.

## 6. What still blocks BSRBR deployment claims

The public benchmark is still a TRAIN-derived **closed-code** diagnostic. Unseen TEST codes are structurally unavailable. A full licensed-MedDRA open-set evaluation requires an authorised terminology resource.

BSRBR transfer still requires:

- real historical BSRBR free text + expert MedDRA reference codes;
- temporal splitting;
- participant-level leakage sensitivity;
- terminology-version and coding-guideline audit;
- target-domain policy calibration using expert-coded BSRBR records;
- locked prospective or later-time confirmation;
- review of explanation usefulness by coding/domain experts;
- governance approval before any external LLM receives governed text.

## 7. Authoritative successful GitHub Actions runs

- v0.1.1 real MedNorm + DeepSeek: `29987846810`
- v0.2 candidate generation: `29991428658`
- v0.2 end-to-end DeepSeek: `29991880549`
- v0.2.1 disjoint policy diagnostic: `29992618418`
- v0.2.2 target-domain policy confirmation: `29993208006`

See `release/v0.2.2.json` and `results/v0.2/` for machine-readable results and diagnostic history.
