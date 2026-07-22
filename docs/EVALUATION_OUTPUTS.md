# Evaluation outputs — v0.0.9

v0.0.9 separates three questions that should not be collapsed into a single accuracy number.

## 1. Can the system retrieve and rank the correct code?

`open_set_metrics.csv` reports exact Accuracy@1 and candidate recall@5/@10 for:

- all TEST records;
- codes already observed in historical TRAIN (`seen_code`);
- codes absent from historical TRAIN (`unseen_code`).

`candidate_retrieval_diagnostics.csv` records the gold candidate rank for each TEST case.

`failure_summary.csv` separates:

- `candidate_generation_failure`: the gold code is absent from the top-10 candidate set;
- `ranking_failure`: the gold code is present but not ranked first;
- `correct`.

This distinction determines the next technical step. Low candidate recall suggests improving terminology coverage, synonyms, or dense retrieval. High candidate recall with low Accuracy@1 suggests improving reranking/disambiguation.

## 2. Does historical expert coding add value?

`historical_memory_value.json` compares the validation-selected system with a pre-specified terminology-only baseline (`history_weight=0`) on the same untouched TEST set.

The comparison is descriptive held-out evaluation. TEST is not used to select the historical-memory weight.

Possible findings include:

- positive Accuracy@1 delta: historical expert coding adds predictive information;
- approximately zero delta: terminology retrieval may already capture most useful information;
- negative delta: historical memory can introduce noise or cohort-specific inconsistency and should not be assumed beneficial.

## 3. How much manual review can be reduced at a required agreement target?

v0.0.9 fixes the AUTO/REVIEW threshold objective. The threshold is selected to **maximise validation coverage subject to a prespecified validation accuracy target**.

`policy_stress_test.csv` repeats this for 90%, 95%, 98%, and 99% targets. Each threshold is selected using validation only and then evaluated unchanged on TEST.

`coverage_accuracy.csv` is a descriptive TEST curve. It must not be used to choose the deployment threshold.

A useful reporting form is:

> At a prespecified 98% validation agreement target, the frozen policy automatically proposed codes for X% of held-out TEST records, with observed TEST agreement Y%; the remaining Z% were routed to human review.

The Results Contract still determines whether the run is reportable. Synthetic, oracle, leaky, or TEST-tuned runs cannot be presented as clinical performance evidence.
