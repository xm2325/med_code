# MedCode — v0.0.10

Research prototype for explainable clinical coding using terminology knowledge and historical expert-coded free text.

## Core question

Can historical expert coding reduce future manual coding workload while keeping agreement with expert reference coding above a prespecified target — **and can every proposed code show the exact supporting text and an auditable explanation?**

The intended transfer setting is a longitudinal cohort such as BSRBR-RA, but the same pipeline can be run against one coding system at a time, for example ICD-10 or MedDRA.

## v0.0.10: code + evidence + why

```text
Clinical record / follow-up text
        ↓
ICD-10 or MedDRA code
        ↓
WHY this code?
        ↓
WHICH exact words support it?
        ↓
terminology knowledge + historical expert-coded support
        ↓
faithfulness audit: retain evidence / remove evidence
        ↓
AUTO_CANDIDATE or HUMAN_REVIEW
```

Every explanation keeps the selected code locked. The explanation layer cannot silently substitute another code.

### Example output shape

```text
Proposed code
MedDRA <code> — Myalgia

Evidence in the record
"severe muscle pain"

Why this code?
The proposed coding label is grounded in the highlighted wording, which overlaps
terminology expressions for Myalgia. Similar historical expert-coded records are
shown separately as supporting provenance.

Faithfulness
- selected-code score on original model input
- selected-code score using evidence only
- selected-code score after evidence removal
```

The actual code/term must come from the supplied terminology resource; the repository does not redistribute a licensed MedDRA dictionary.

## Explainability design

v0.0.10 is informed by Mingyang Li's Manchester work on explainable ICD coding and the EACL 2026 paper *Evaluation and LLM-Guided Learning of ICD Coding Rationales*.

The project keeps two questions separate:

- **faithfulness** — does the highlighted evidence actually affect the model's selected-code score?
- **plausibility** — would a clinical expert judge the evidence/rationale as appropriate?

The system therefore writes both perturbation-style faithfulness diagnostics and a `rationale_review_template.csv` for expert plausibility review.

See `docs/EXPLAINABILITY.md` for the full design and methodological caveats.

## Optional DeepSeek rationale generation

`DEEPSEEK_API_KEY` is read from the environment when the optional DeepSeek explanation path is used.

DeepSeek is used as a **locked-code rationale writer**, not as an unrestricted coding agent:

- it receives the already-selected code;
- it receives only approved verbatim evidence spans plus terminology knowledge by default;
- it cannot change the code;
- quoted evidence must exactly match the allowed source text;
- invalid output is rejected and falls back to the deterministic grounded explanation.

External calls require explicit `--allow-external-llm` and are blocked for `restricted` or `private` clinical data. For BSRBR or other governed cohorts, use an approved/local model endpoint unless external processing is explicitly permitted.

## Quick start

```bash
python -m pip install -e .
pytest -q
python scripts/run_demo.py
```

## Real-data benchmark

```bash
python scripts/prepare_cadec_public.py \
  --cadec-root /path/to/CADEC \
  --output data/processed/cadec.csv

python scripts/split_real_dataset.py \
  --records data/processed/cadec.csv \
  --output data/processed/cadec_with_split.csv

python scripts/run_real_benchmark.py \
  --records data/processed/cadec_with_split.csv \
  --terminology /secure/path/meddra_candidates.csv \
  --coding-system MedDRA \
  --output-dir outputs/cadec_v0010 \
  --reference-labels-external
```

Then generate grounded explanations for the untouched TEST predictions:

```bash
python scripts/explain_benchmark.py \
  --records data/processed/cadec_with_split.csv \
  --terminology /secure/path/meddra_candidates.csv \
  --coding-system MedDRA \
  --benchmark-dir outputs/cadec_v0010
```

For public/synthetic data only, optional DeepSeek rationales can be generated with:

```bash
python scripts/explain_benchmark.py \
  --records data/processed/cadec_with_split.csv \
  --terminology /secure/path/meddra_candidates.csv \
  --coding-system MedDRA \
  --benchmark-dir outputs/cadec_v0010 \
  --deepseek \
  --allow-external-llm \
  --data-classification public
```

## New uncoded records: prediction + explanation

```bash
python scripts/predict_explain_uncoded.py \
  --historical historical_train.csv \
  --terminology /secure/path/meddra_candidates.csv \
  --coding-system MedDRA \
  --input new_uncoded.csv \
  --frozen-policy outputs/cadec_v0010/frozen_policy.json \
  --output-dir outputs/new_records
```

This writes the code proposal, confidence, AUTO/REVIEW decision, exact evidence spans, explanation, terminology support, historical support, and faithfulness diagnostics.

## Terminology schema

Recommended columns:

```text
system,code,term,synonyms,definition,hierarchy,knowledge_source
```

`term` stays clean for display. Synonyms and definitions are added only to a separate retrieval `search_text` representation.

One benchmark run targets one coding system. Filter ICD-10 and MedDRA into separate runs rather than mixing their code spaces.

## Main outputs

Benchmark outputs include:

- `metrics.json`
- `predictions.csv`
- `model_selection.csv`
- `results_contract.json`
- `frozen_policy.json`
- `open_set_metrics.csv`
- `candidate_retrieval_diagnostics.csv`
- `historical_memory_value.json`
- `coverage_accuracy.png`
- `policy_workload.png`

Explainability outputs include:

- `explanations.csv`
- `explanations.jsonl`
- `explanations.html` — record-level view with highlighted evidence
- `explainability_metrics.json`
- `rationale_review_template.csv`

## Evidence boundary

A benchmark result is reportable only when the Results Contract confirms external human reference labels, group-disjoint held-out TEST data, no TEST-derived terminology leakage, no TEST tuning, non-synthetic data, and recorded provenance.

A fluent LLM rationale does not make an incorrect code correct. Code accuracy, rationale faithfulness, and expert plausibility must be evaluated separately.

## Data and terminology

Do not commit raw participant-level data, processed sensitive data, or a full licensed MedDRA distribution. Supply terminology locally under the applicable licence.

This repository is a research coding-support prototype, not a clinical decision system.
