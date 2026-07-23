# MIPA local phenotyping acceptance protocol

This is the v0.3.0 bridge from the public MIPA feasibility audit to authorised note-level phenotype experiments.

## Governance boundary

`run_mipa_local_phenotyping.py` is evaluation-only and performs no model, API, or network calls. The authorised MIPA/MIMIC note file and locally generated predictions stay on the local filesystem. Remote URL-like inputs are rejected by the CLI.

Do not commit restricted discharge summaries, model prompts containing restricted text, or unrestricted note-level outputs to this repository.

## First six RA-relevant phenotypes

The default evaluation set is:

- `hypertension`
- `depression`
- `diabetes_type_2`
- `hfpef`
- `vte_past`
- `obesity`

These are a methods pilot, not an RA prevalence panel.

## Required local inputs

### Labels

Authorised/local copy of MIPA `golden_labels.csv`. Required columns include `note_id`, `subject_id`, and the phenotype columns.

### Notes

Authorised/local `golden_notes.csv`. The evaluator joins on `note_id` when available, otherwise `hadm_id`. It accepts one of `text`, `note_text`, or `discharge_summary` as the note-text column.

### Predictions

CSV or JSONL with one row per note-phenotype pair:

```text
note_id,phenotype,prediction,evidence,assertion,temporality
```

Accepted binary prediction values include `0/1`, `absent/present`, `negative/positive`, and `no/yes`.

Assertion contract:

```text
present | absent | possible | negated | family_history | unknown
```

Temporality contract:

```text
current | historical | resolved | unknown
```

The `evidence` field should be an exact source span from the note for positive predictions.

## Subject-disjoint split

The runner deterministically assigns every `subject_id` to exactly one of train, validation, or test using a salted SHA-256 hash. Repeated admissions/notes for one subject therefore stay in the same split.

The split manifest is always written and overlap counts must be zero.

## Run

```bash
python scripts/run_mipa_local_phenotyping.py \
  --labels /secure/path/golden_labels.csv \
  --notes /secure/path/golden_notes.csv \
  --predictions /secure/path/local_model_predictions.jsonl \
  --output-dir /secure/path/mipa_eval
```

For a frozen confirmatory test split:

```bash
python scripts/run_mipa_local_phenotyping.py \
  --labels /secure/path/golden_labels.csv \
  --notes /secure/path/golden_notes.csv \
  --predictions /secure/path/local_model_predictions.jsonl \
  --evaluation-scope test \
  --output-dir /secure/path/mipa_eval_test
```

## Automated Stage 1/2 gate

Default gate:

- macro-F1 >= 0.85
- at least 5 of 6 phenotypes have F1 >= 0.80
- no common phenotype (default: at least 20 gold positives) has recall < 0.60
- prediction coverage = 100%
- prediction/assertion/temporality contract has no invalid records
- positive-prediction verbatim evidence rate >= 0.99
- subject overlap across train/validation/test = 0

A failure produces `FAIL_AUTOMATED_GATE`.

Passing all automated checks without human evidence review produces:

```text
PASS_AUTOMATED_PENDING_HUMAN_EVIDENCE_AUDIT
```

This is not a final scientific PASS.

## Human evidence gate

MIPA public phenotype labels do not contain gold assertion or temporality labels. Therefore the evaluator does **not** claim assertion/temporality accuracy from MIPA.

A separate reviewer audit can be supplied as:

```text
note_id,phenotype,supports_prediction,severe_context_error
```

Default final gate:

- evidence clinically supports prediction >= 90%
- severe context error rate < 5%

Severe context errors include negation mistakes, family-history-as-patient mistakes, and unsupported possible-to-confirmed promotion.

With an audit file:

```bash
python scripts/run_mipa_local_phenotyping.py \
  --labels /secure/path/golden_labels.csv \
  --notes /secure/path/golden_notes.csv \
  --predictions /secure/path/local_model_predictions.jsonl \
  --evidence-audit /secure/path/evidence_audit.csv \
  --output-dir /secure/path/mipa_eval \
  --require-final-pass
```

`--require-final-pass` exits non-zero unless the final status is `PASS`.

## Outputs

- `summary.json`: governance flags, split audit, metrics, evidence checks, explicit gate status
- `phenotype_metrics.csv`: per-phenotype confusion counts, precision, recall, specificity, F1
- `error_cases.csv`: missing/invalid predictions, classification errors, non-verbatim evidence, contract failures
- `subject_split_manifest.csv`: note/admission/subject split assignment

The evaluator intentionally does not write full note text into these outputs.

## Scientific progression

A final PASS here supports moving to the next experiment: clinician gold × text model × structured coding discordance. It does not itself demonstrate under-recording, hidden comorbidity, RA population prevalence, or improved downstream RA outcomes.
