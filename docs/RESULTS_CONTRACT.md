# Results Contract

A MedCode benchmark is **reportable** only when all of the following are true:

1. Evaluation uses externally supplied human reference labels.
2. TEST records are group-disjoint from TRAIN and validation data.
3. TEST labels are not used to construct the default terminology candidate vocabulary.
4. Model selection, confidence calibration, and AUTO/REVIEW policy selection do not use TEST labels.
5. The result is not from bundled synthetic smoke-test data.
6. Dataset and terminology provenance/fingerprints are recorded.

Runs that fail any condition must be labelled as `synthetic_smoke_test`, `development_only`, `oracle_diagnostic`, or `non_reportable`.

## Required headline outputs

For a reportable benchmark, provide at minimum Accuracy@1 and Accuracy@5, candidate recall@5 and recall@10, seen-code versus unseen-code Accuracy@1, calibration metrics, coverage/accuracy results, AUTO coverage at prespecified agreement targets, document-level bootstrap confidence intervals, and leakage/reference-label audit summaries.

## Selective coding claim

Do not state that a percentage of records can be automatically coded unless the AUTO/REVIEW policy was frozen without TEST labels and held-out TEST performance meets the stated agreement target. Otherwise describe records as candidates for automated coding under a validation-derived policy.
