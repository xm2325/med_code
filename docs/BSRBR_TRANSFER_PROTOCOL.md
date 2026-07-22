# BSRBR Transfer Protocol

## Objective

Evaluate whether historical expert-coded adverse-event/comorbidity free text can support future coding while preserving a prespecified agreement target and reducing manual review.

## Minimum schema

Each historical record should ideally contain a stable record identifier, participant identifier, report/follow-up date, raw text, codable mention or event text when available, historical terminology code, historical terminology term, and coding/version metadata when available.

## Primary split

Use a temporal design rather than a random record split:

```text
Historical period -> TRAIN
Intermediate period -> VALIDATION
Later expert-coded period -> TEST
Future uncoded period -> deployment-only inference
```

The TEST period must remain untouched during model selection, confidence calibration, and AUTO/REVIEW threshold selection.

## Leakage controls

Split at source-document level at minimum. Run a participant-level sensitivity analysis where repeated participants may otherwise cross periods. Audit exact text overlap, repeated mentions, duplicated records, and label-version changes.

## Reference-label audit

Historical codes are a reference standard, not assumed perfect truth. Check identical/similar surfaces assigned to different codes, one code represented by inconsistent term strings, changes in coding practice over time, and terminology-version changes.

## Evaluation

Report exact-code Accuracy@1 and top-k retrieval metrics where appropriate, terminology/hierarchy-level agreement when available, seen versus unseen historical codes, candidate-recall versus reranking failures, calibration, coverage-accuracy/risk-coverage curves, and document-level bootstrap confidence intervals.

## Selective coding

Choose the AUTO/REVIEW policy using validation data only. Report the fraction of records that can be automatically proposed at prespecified agreement targets such as 95%, 98%, and 99%. New data must use the frozen retrieval/reranking model, confidence calibrator, and threshold.

## Prospective check

Before operational use, obtain a small newly double-coded sample. Prefer two independent coders with adjudication for disagreements. Compare the frozen system against this prospective reference sample and re-check calibration and subgroup failure patterns.

## Safety boundary

The system is a research coding-support tool. Low-confidence, ambiguous, novel, conflicting, or clinically complex records should route to expert review rather than receive unsupported automatic acceptance.
