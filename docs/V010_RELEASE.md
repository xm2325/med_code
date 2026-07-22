# MedCode v0.1.0 release guide

v0.1.0 is the first software-complete application release.

## End-to-end contract

```text
clinical text
→ frozen candidate generation/ranking
→ uncertainty and OOD checks
→ AUTO_CANDIDATE / TOP_K_HUMAN_CHOICE / FULL_EXPERT_REVIEW
→ candidate-specific exact evidence + rationale
→ persistent human review
→ versioned feedback ledger
→ future-release memory only
→ replayable audit bundle
```

## Required software capabilities

- audited real-data adapter;
- held-out evaluation;
- Results Contract;
- frozen policy;
- uncertainty-aware routing;
- grounded Top-K rationale objects;
- persistent expert review;
- expert feedback ledger;
- replayable audit trail.

## Release checklist for an actual model/data candidate

A software release being complete is not enough. A specific experiment/model candidate should also have:

1. dataset audit passed;
2. data/source versions recorded;
3. external human reference labels declared;
4. group/patient leakage checks passed;
5. TEST untouched during model, prompt and threshold selection;
6. candidate terminology provenance recorded;
7. coding metrics and candidate Recall@K reported;
8. uncertainty/workload metrics reported;
9. explanation quality and expert plausibility review reported;
10. frozen policy, Results Contract and audit bundle generated.

## Safety invariant

Explanation, uncertainty, OOD and human-review layers can make a decision more conservative. They must not silently promote a case from review to automatic handling.
