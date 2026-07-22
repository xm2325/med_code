# MedCode — v0.0.11

Research prototype for **explainable clinical coding** using terminology knowledge and historical expert-coded text.

The desired output is not only a code:

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

## v0.0.11: two real benchmark profiles

v0.0.11 makes an important distinction between two different clinical coding tasks.

### A. CADEC → MedDRA

```text
patient-authored post
    -> annotated adverse-event mention
    -> one MedDRA concept
    -> evidence + rationale
```

Task type: **single-label concept normalization**.

Primary metrics include Accuracy@1/5, candidate recall, and validation-selected AUTO/HUMAN_REVIEW performance.

Split unit: source document/post.

### B. MIMIC-IV-Note → ICD-10

```text
discharge summary
    -> multiple ICD-10 diagnosis codes
    -> one evidence/rationale object per proposed code
```

Task type: **multi-label document coding**.

Primary metrics include micro/macro F1, precision/recall@k, seen/unseen-code recall, and a validation-selected per-code proposal policy.

Split unit: `subject_id`. All admissions for one patient stay in one split.

**Do not compare CADEC Accuracy@1 directly with MIMIC micro-F1 as if they were the same task.** `compare_benchmarks.py` creates a side-by-side report but intentionally does not compute a pooled score.

See `docs/BENCHMARKS.md`.

## Explainability

For every proposed code, MedCode can emit:

```text
Code
ICD-10 I21.4 — Non-ST elevation myocardial infarction

Evidence
"NSTEMI"
"troponin elevation with ischemic changes"

Why this code?
The proposed code is supported by the highlighted source wording and the supplied
terminology knowledge. Similar historical expert-coded examples are shown separately.

External knowledge
preferred term / synonyms / definition / hierarchy / knowledge source

Historical support
similar TRAIN records carrying the same code

Faithfulness audit
- selected-code score on the original model input
- selected-code score with evidence only
- selected-code score after evidence removal
```

Evidence spans are copied verbatim from the source text with character offsets.

v0.0.11 also supports comparison with human rationale annotations using:

```text
record_id,code,start,end,quote
```

and reports character-level rationale precision, recall, and F1 by record-code pair.

The design is informed by Mingyang Li's Manchester work and the EACL 2026 paper *Evaluation and LLM-Guided Learning of ICD Coding Rationales*: **faithfulness** and **plausibility** remain separate evaluation questions.

See `docs/EXPLAINABILITY.md` and `docs/BENCHMARKS.md`.

## Optional DeepSeek rationale writer

`DEEPSEEK_API_KEY` is read from the environment only when the optional LLM path is explicitly enabled.

DeepSeek is used as a **locked-code rationale writer**, not as an unrestricted coding agent:

- the code is already selected before the call;
- only approved evidence spans and terminology knowledge are supplied by default;
- the returned code must equal the locked code;
- returned evidence quotes must match the approved verbatim source spans;
- invalid output is rejected and deterministic grounded output is retained.

External calls require explicit opt-in and are blocked by the client for `restricted` or `private` clinical data. Do not send MIMIC, BSRBR, or other governed clinical text to an external endpoint unless the applicable governance explicitly permits it.

## Quick start

```bash
python -m pip install -e .
pytest -q
python scripts/run_explainable_demo.py
```

## Benchmark A — CADEC → MedDRA

Prepare CADEC:

```bash
python scripts/prepare_cadec_public.py \
  --cadec-root /path/to/CADEC \
  --output data/processed/cadec.csv

python scripts/split_real_dataset.py \
  --records data/processed/cadec.csv \
  --output data/processed/cadec_with_split.csv
```

Run the single-label MedDRA benchmark using an authorised terminology resource:

```bash
python scripts/run_real_benchmark.py \
  --records data/processed/cadec_with_split.csv \
  --terminology /secure/path/meddra_candidates.csv \
  --coding-system MedDRA \
  --benchmark-profile cadec_meddra_normalization \
  --output-dir outputs/cadec_v0011 \
  --reference-labels-external
```

Generate evidence-grounded explanations:

```bash
python scripts/explain_benchmark.py \
  --records data/processed/cadec_with_split.csv \
  --terminology /secure/path/meddra_candidates.csv \
  --coding-system MedDRA \
  --benchmark-dir outputs/cadec_v0011
```

## Benchmark B — MIMIC-IV-Note → ICD-10

MIMIC-IV-Note is credentialed-access PhysioNet data. Keep all source and derived patient-level files outside this public repository.

A reproducible default is to use MIMIC-IV-Note v2.2 with the corresponding MIMIC-IV v2.2 diagnosis tables and record the exact versions in the experiment manifest.

Prepare one multi-label record per hospitalization and split by patient:

```bash
python scripts/prepare_mimic_iv_icd10.py \
  --discharge /secure/mimic-iv-note/2.2/note/discharge.csv.gz \
  --diagnoses /secure/mimiciv/2.2/hosp/diagnoses_icd.csv.gz \
  --dictionary /secure/mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz \
  --output /secure/derived/mimic_icd10_records.csv

python scripts/prepare_icd10_terminology.py \
  --dictionary /secure/mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz \
  --output /secure/derived/icd10_terminology.csv
```

Run the multi-label benchmark:

```bash
python scripts/run_mimic_icd10_benchmark.py \
  --records /secure/derived/mimic_icd10_records.csv \
  --terminology /secure/derived/icd10_terminology.csv \
  --output-dir /secure/results/mimic_v0011 \
  --target-proposal-precision 0.95 \
  --reference-labels-external
```

Generate **one explanation per proposed ICD code**:

```bash
python scripts/explain_mimic_icd10.py \
  --records /secure/derived/mimic_icd10_records.csv \
  --terminology /secure/derived/icd10_terminology.csv \
  --benchmark-dir /secure/results/mimic_v0011
```

For MIMIC, `target-proposal-precision=0.95` means a target precision for **individual code proposals**. It does not mean that 95% of entire discharge summaries can be completely auto-coded.

## Human rationale evaluation

When governed human rationale annotations are available:

```bash
python scripts/evaluate_rationale_annotations.py \
  --explanations /secure/results/mimic_v0011/explainability/explanations.csv \
  --reference /secure/rationales/reference_spans.csv \
  --records /secure/derived/mimic_icd10_records.csv \
  --output-dir /secure/results/mimic_v0011/rationale_eval
```

The evaluator first audits source offsets/quotes, then reports record-code character-level overlap metrics.

A fluent LLM rationale does not make an incorrect code correct. Evaluate separately:

1. **coding performance**;
2. **evidence faithfulness**;
3. **human/expert plausibility**.

## Side-by-side benchmark summary

```bash
python scripts/compare_benchmarks.py \
  --cadec-dir outputs/cadec_v0011 \
  --mimic-dir /secure/results/mimic_v0011 \
  --output-dir outputs/dual_benchmark_v0011
```

Outputs include `dual_benchmark_summary.html`, but no pooled score is calculated across the incompatible tasks.

## Terminology schema

Recommended columns:

```text
system,code,term,synonyms,definition,hierarchy,knowledge_source
```

`term` stays human-readable. Synonyms and definitions can contribute to a separate retrieval representation. One benchmark run targets one coding system; do not mix ICD-10 and MedDRA code spaces in one candidate dictionary.

## Evidence boundary

A benchmark is marked reportable only when the Results Contract confirms the required conditions: external reference labels, group-disjoint held-out TEST data, no TEST-derived terminology leakage, no TEST tuning, non-synthetic data, and recorded provenance.

Synthetic smoke tests, oracle diagnostics, and TEST-tuned analyses must not be presented as clinical performance evidence.

## Data governance

Do not commit:

- raw participant/patient-level clinical data;
- MIMIC notes or derived patient-level datasets;
- governed rationale annotations derived from MIMIC;
- BSRBR free text or restricted cohort extracts;
- a full licensed MedDRA distribution;
- API keys or secrets.

This repository is a research coding-support prototype, not a clinical decision system.
