# Real benchmark readiness — v0.0.12

A model metric is only meaningful when the data path that produced the evaluation examples is itself auditable. v0.0.12 therefore adds explicit pre-flight gates before the main CADEC and MIMIC benchmarks.

## CADEC -> MedDRA

### Why the parser is part of the scientific method

CADEC uses BRAT-style annotations. A text-bound annotation may contain one contiguous span or several discontinuous spans.

Example:

```text
T2  ADE 6 10;19 23  pain legs
```

The correct representation is two source intervals. Converting it to one outer interval `6-23` would include text that was never annotated and would make later evidence highlighting misleading.

v0.0.12 stores:

```text
annotation_id
normalization_id
start
end
spans_json
is_discontinuous
offset_text
offset_match
```

`start/end` are only convenience outer bounds. `spans_json` is the authoritative source-span representation for discontinuous annotations.

### CADEC readiness gate

The default audit checks:

- every record has a reference MedDRA code;
- BRAT source offsets reproduce the annotation text at >=99% rate by default;
- the supplied terminology contains >=95% of unique reference codes by default;
- duplicate rows and multiple codes linked to the same annotation are surfaced;
- discontinuous annotations are counted and preferentially sampled for manual review.

The output is:

```text
audit/dataset_audit.json
audit/manual_parser_review_sample.csv
```

A failed hard check blocks the one-command pipeline unless `--allow-audit-failures` is supplied explicitly for debugging. Overriding the gate does not turn a questionable run into reportable evidence.

### Task input span vs rationale label

In CADEC normalization, the adverse-event mention is already part of the task input. v0.0.12 preserves its exact BRAT location and labels it `task_input_span` in the explanation layer.

This is not a gold-code rationale: no reference MedDRA code is used to choose the location. The distinction matters when the same phrase appears more than once in one post.

## MIMIC-IV-Note -> ICD-10

The MIMIC path audits prepared benchmark records before model fitting.

Checks include:

- `subject_id` has zero overlap across train/validation/test;
- no empty clinical text records;
- no records with an empty ICD-10 label set;
- external ICD-10 terminology coverage >=99% by default;
- split-specific subject and record counts;
- note-length character quantiles;
- codes-per-note quantiles.

Outputs:

```text
audit/dataset_audit.json
audit/manual_data_review_sample.csv
```

The manual sample deliberately includes very long notes and high-code-count records before random controls, because those cases are operationally more likely to reveal data-joining or truncation problems.

## Case-level expert review

A real benchmark should not stop at aggregate metrics. The CADEC one-command runner creates a review casebook that prioritizes:

1. AUTO_CANDIDATE errors;
2. high-confidence errors;
3. unseen-code errors;
4. insufficient-evidence cases;
5. other errors;
6. correct controls.

The casebook includes the original context, task mention, gold/predicted code, confidence, decision, explanation, evidence and candidate ranking. Blank expert fields support structured review.

## Evidence hierarchy

Use the following interpretation order:

```text
parser integrity
    ↓
dataset/terminology readiness
    ↓
split leakage audit
    ↓
held-out coding metrics
    ↓
evidence faithfulness
    ↓
expert plausibility
    ↓
workload / selective-review claim
```

A later layer cannot repair a failure in an earlier layer. For example, a fluent LLM rationale cannot rescue a wrongly aligned source span or a TEST-tuned policy.

## One-command runners

CADEC:

```bash
python scripts/run_cadec_v0012.py \
  --cadec-root /path/to/CADEC \
  --terminology /secure/meddra.csv \
  --output-dir outputs/cadec_v0012
```

MIMIC:

```bash
python scripts/run_mimic_v0012.py \
  --records /secure/derived/mimic_icd10_records.csv \
  --terminology /secure/derived/icd10_terminology.csv \
  --output-dir /secure/results/mimic_v0012
```

These runners do not download or redistribute governed/licensed datasets or terminology resources.
