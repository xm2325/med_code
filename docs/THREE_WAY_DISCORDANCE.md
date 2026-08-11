# Gold × Text × Structured discordance protocol

This stage asks whether clinically supported comorbidity information in free text adds ascertainment beyond structured records.

## Inputs

- **G (gold):** clinician/expert phenotype label.
- **T (text):** phenotype prediction from the Stage 1/2 evidence-grounded text model.
- **C (structured):** phenotype indicator derived from a reviewed structured-code definition.

The core candidate recovery cell is:

```text
G=1, T=1, C=0
```

This means the phenotype is gold-positive, the text model identifies it, and the structured representation does not.

## Run

```bash
python scripts/run_three_way_discordance.py \
  --labels /secure/path/golden_labels.csv \
  --text-predictions /secure/path/local_model_predictions.csv \
  --structured-status /secure/path/structured_status.csv \
  --upstream-summary /secure/path/mipa_eval/summary.json \
  --structured-scope-validated \
  --output-dir /secure/path/discordance
```

`structured_status.csv` has one row per note-phenotype pair:

```text
note_id,phenotype,structured_positive
```

## Metrics

For each phenotype and overall, the evaluator reports:

- structured sensitivity among gold-positive observations
- text sensitivity among gold-positive observations
- combined `(T or C)` sensitivity among gold-positive observations
- number of `G=1,C=0` gold-positive observations missed by structured data
- number of `G=1,T=1,C=0` observations recovered by text
- recoverable code-missed fraction: `N(G=1,T=1,C=0) / N(G=1,C=0)`
- gold PPV among `T=1,C=0` observations
- all eight G/T/C cells

## Interpretation gate

The tool allows confirmatory under-recording language only when all three conditions hold:

1. Stage 1/2 `summary.json` has `final_status = PASS`.
2. The structured phenotype definition has been reviewed and `--structured-scope-validated` is supplied.
3. Text and structured inputs have 100% expected note-phenotype coverage.

Otherwise the result is explicitly labelled:

```text
EXPLORATORY_DISCORDANCE_ONLY
```

A low or zero under-recording rate is not a pipeline failure. It is a valid scientific result if the inputs and gates are valid.

## Important boundary

This evaluator does not turn every `T=1,C=0` observation into a hidden comorbidity. Gold support is required for the confirmatory MIPA analysis. In a real BSRBR-RA cohort without note-level clinician gold for every patient, an expert-validation design is needed before analogous text-positive/code-absent cases can be called validated under-recording.
