# MedCode — v0.0.12

Research prototype for **explainable clinical coding** using terminology knowledge and historical expert-coded text.

```text
clinical text
    ↓
ICD-10 / MedDRA code
    ↓
which exact words support it?
    ↓
why do those words support this code?
    ↓
terminology knowledge + historical expert-coded provenance
    ↓
faithfulness / plausibility checks
    ↓
AUTO_CANDIDATE, CODE_PROPOSAL, or HUMAN_REVIEW
```

The explanation layer keeps the selected code locked. It cannot silently replace the code or invent evidence that is absent from the source text.

## What v0.0.12 changes

v0.0.12 focuses on **real-benchmark readiness**, not another model layer.

Before a benchmark is allowed to run, the data path can now audit the inputs that determine whether the later explanation and performance numbers are trustworthy.

### CADEC → MedDRA

The parser now preserves the original BRAT annotation structure:

```text
source document
    ↓
T annotation + exact character span(s)
    ↓
MedDRA normalization
    ↓
offset/text integrity audit
    ↓
terminology coverage audit
    ↓
document-level TRAIN / VAL / TEST
    ↓
held-out coding benchmark
    ↓
exact task-input evidence + why-this-code explanation
    ↓
priority clinical/coder review casebook
```

Discontinuous spans such as `6 10;19 23` remain two spans. They are not collapsed into the outer interval `6–23`, which would incorrectly highlight intervening text.

The parser writes `spans_json`, `is_discontinuous`, `offset_text`, and `offset_match`. A benchmark can be blocked before model fitting when source offsets are unreliable or the supplied terminology does not cover enough reference codes.

### MIMIC-IV-Note → ICD-10

The multi-label path remains patient-disjoint and now has a pre-flight audit:

```text
prepared discharge summaries + ICD-10 labels
    ↓
subject_id leakage check
    ↓
empty text / empty label check
    ↓
ICD-10 terminology coverage
    ↓
note length + codes-per-note distribution
    ↓
multi-label benchmark
    ↓
one explanation object per proposed ICD code
```

A `target_proposal_precision=0.95` policy applies to **individual code proposals**. It must not be described as 95% of whole discharge summaries being fully auto-coded.

## Two benchmark profiles

### A. CADEC → MedDRA

Task type: **single-label concept normalization**.

Primary evaluation includes Accuracy@1/5, candidate recall, seen/unseen-code analysis, historical-memory ablation, and validation-selected AUTO/HUMAN_REVIEW policy performance.

Split unit: source document/post.

### B. MIMIC-IV-Note → ICD-10

Task type: **multi-label document coding**.

Primary evaluation includes micro/macro F1, precision/recall@k, seen/unseen-code recall, historical-memory ablation, and a validation-selected per-code proposal policy.

Split unit: `subject_id`; all admissions for one patient remain in one split.

Do not pool CADEC Accuracy@1 and MIMIC micro-F1 into a single score. They are different tasks.

## Explainability

For each proposed code the system can show:

```text
Code
MedDRA / ICD-10 code + preferred term

Evidence
exact verbatim source span(s) with character offsets

Why this code?
terminology mapping + historical coding support

External knowledge
preferred term / synonyms / definition / hierarchy / knowledge source

Historical support
similar TRAIN records carrying the same code

Faithfulness
selected-code score on original input
selected-code score with evidence only
selected-code score after evidence removal
```

For CADEC, v0.0.12 prefers the exact task-provided BRAT mention spans over re-searching the text. This preserves the intended mention location when the same phrase appears multiple times.

The design keeps three questions separate:

1. **coding accuracy** — is the code correct?
2. **faithfulness** — does the highlighted evidence actually affect the model score?
3. **plausibility** — would a clinician/coder judge the evidence and explanation appropriate?

## Clinical review casebook

The CADEC end-to-end runner creates a casebook that prioritizes:

- AUTO_CANDIDATE errors;
- high-confidence errors;
- unseen-code errors;
- insufficient-evidence cases;
- other errors;
- correct controls.

The HTML/CSV includes original source context, task mention, prediction, confidence, rationale, evidence and candidate list, plus blank expert-review fields.

## Optional DeepSeek rationale writer

`DEEPSEEK_API_KEY` is read from the environment only when the optional LLM path is explicitly enabled.

DeepSeek is a **locked-code rationale writer**, not an unrestricted coding agent:

- the code is selected before the call;
- only approved evidence spans and terminology knowledge are supplied by default;
- the returned code must equal the locked code;
- returned evidence quotes must match approved verbatim source spans;
- invalid output is rejected;
- no affirmative grounded evidence means no LLM call.

External calls are blocked by the client for `restricted` or `private` clinical data. Do not send MIMIC, BSRBR, or other governed clinical text to an external endpoint unless the applicable governance explicitly permits it.

## Quick start

```bash
python -m pip install -e .
pytest -q
python scripts/run_explainable_demo.py
python scripts/run_multilabel_synthetic_demo.py
```

## One-command CADEC v0.0.12 path

Use an authorised terminology file with columns such as:

```text
system,code,term,synonyms,definition,hierarchy,knowledge_source
```

Then run:

```bash
python scripts/run_cadec_v0012.py \
  --cadec-root /path/to/CADEC \
  --terminology /secure/path/meddra_candidates.csv \
  --output-dir outputs/cadec_v0012 \
  --target-auto-accuracy 0.95
```

Main outputs:

```text
outputs/cadec_v0012/
├── data/
│   ├── cadec_parsed.csv
│   └── cadec_split.csv
├── audit/
│   ├── dataset_audit.json
│   └── manual_parser_review_sample.csv
├── benchmark/
│   ├── metrics.json
│   ├── predictions.csv
│   ├── results_contract.json
│   ├── frozen_policy.json
│   ├── explanations.html
│   └── ...
├── casebook/
│   ├── review_casebook.csv
│   └── review_casebook.html
└── pipeline_summary.json
```

The pipeline stops before benchmarking when the default data-readiness gate fails, unless `--allow-audit-failures` is explicitly supplied for debugging.

Individual steps are also available:

```bash
python scripts/prepare_cadec_public.py --cadec-root /path/to/CADEC --output data/cadec.csv
python scripts/audit_cadec_dataset.py --records data/cadec.csv --terminology /secure/meddra.csv --output-dir outputs/audit
```

## One-command MIMIC v0.0.12 path

MIMIC-IV-Note is credentialed-access PhysioNet data. Keep source and derived patient-level files outside this public repository.

First prepare records and terminology using the v0.0.11 adapters, then run:

```bash
python scripts/run_mimic_v0012.py \
  --records /secure/derived/mimic_icd10_records.csv \
  --terminology /secure/derived/icd10_terminology.csv \
  --output-dir /secure/results/mimic_v0012 \
  --target-proposal-precision 0.95
```

The pre-flight audit checks patient leakage, empty records/labels, terminology coverage and dataset distributions before model fitting.

## Human rationale evaluation

When governed human rationale annotations are available:

```bash
python scripts/evaluate_rationale_annotations.py \
  --explanations /secure/results/mimic_v0012/benchmark/explainability/explanations.csv \
  --reference /secure/rationales/reference_spans.csv \
  --records /secure/derived/mimic_icd10_records.csv \
  --output-dir /secure/results/mimic_v0012/rationale_eval
```

Reference schema:

```text
record_id,code,start,end,quote
```

The evaluator audits source offsets/quotes before computing record-code character-level rationale precision, recall and F1.

## Results boundary

A benchmark is marked reportable only when the Results Contract confirms the required conditions: external reference labels, group-disjoint held-out TEST data, no TEST-derived terminology leakage, no TEST tuning, non-synthetic data, and recorded provenance.

A fluent explanation does not make an incorrect code correct. Synthetic smoke tests, oracle diagnostics and TEST-tuned analyses must not be presented as clinical performance evidence.

## Data governance

Do not commit:

- raw participant/patient-level clinical data;
- MIMIC notes or derived patient-level datasets;
- governed rationale annotations;
- BSRBR free text or restricted cohort extracts;
- a full licensed MedDRA distribution;
- API keys or secrets.

See `docs/BENCHMARKS.md`, `docs/EXPLAINABILITY.md`, and `docs/BSRBR_TRANSFER_PROTOCOL.md` for the study design boundaries.
