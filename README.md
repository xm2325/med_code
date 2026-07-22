# MedCode — v0.0.13

Research prototype for **explainable clinical coding** using terminology knowledge, historical expert-coded records, stronger biomedical retrieval/reranking, and auditable evidence.

```text
clinical text
    ↓
ICD-10 / MedDRA code candidates
    ↓
lexical terminology + historical coding memory
    ↓
optional biomedical dense retrieval
    ↓
optional cross-encoder / frozen-candidate LLM reranking
    ↓
selected code
    ↓
which exact words support it?
    ↓
why do those words support this code?
    ↓
faithfulness + clinical context + explanation quality gate
    ↓
AUTO_CANDIDATE / CODE_PROPOSAL / HUMAN_REVIEW
```

The core rule remains: **a stronger model does not get a weaker evidence standard**. A model may improve candidate retrieval or ranking, but the final code still has to pass the same source-evidence and review safeguards.

## v0.0.13: model quality without weakening auditability

### 1. Biomedical dense retrieval

An optional sentence-transformer-compatible model can add semantic similarity between:

- the clinical mention and terminology concepts;
- the clinical mention and historical expert-coded examples, aggregated by code.

The repository does not bundle model weights or silently download them from the core package. Supply a model name or local path explicitly with `--dense-model`.

### 2. Cross-encoder reranking

An optional cross-encoder can rerank a **frozen candidate pool**. It cannot create a code outside the supplied terminology dictionary.

### 3. Validation-only staged model selection

The advanced CADEC benchmark does not assume a larger model is better:

```text
TRAIN
  ↓
build retrieval/indexes

VALIDATION
  1. choose historical-memory weight
  2. decide whether dense fusion improves results
  3. decide whether cross-encoder reranking improves results
  4. break ties in favour of the simpler model

TEST
  ↓
evaluate the frozen final configuration once
```

The selected configuration is written to `frozen_policy.json` and can be rebuilt for explanation and future inference.

### 4. DeepSeek has two separate roles

**Candidate reranker**

```text
frozen Top-N codes
    ↓
DeepSeek fixed-prompt reranking
    ↓
exact same code set, different order only
```

The output is rejected if a code is added, removed, or duplicated.

**Rationale writer**

```text
selected code already locked
    +
approved verbatim evidence
    ↓
why-this-code explanation
```

The rationale writer cannot change the code or manufacture new evidence.

External DeepSeek calls require explicit opt-in and remain blocked for `restricted` or `private` clinical data in the provided client.

### 5. Explanation quality gate

After coding and explanation generation, every explanation is classified as `PASS`, `WARN`, or `FAIL` using auditable checks such as:

- evidence exists;
- quotes are verbatim in the source;
- obvious negation/uncertainty/family-history context does not require review;
- terminology support is present;
- retain/remove faithfulness diagnostics are available where supported.

A failed explanation can only make the operational decision more conservative:

```text
AUTO_CANDIDATE / CODE_PROPOSAL
        ↓
HUMAN_REVIEW
```

It can never promote a review case to automatic handling.

## Benchmark A — CADEC → MedDRA

Task type: **single-label concept normalization**.

Primary evaluation:

- Accuracy@1 / Accuracy@5;
- candidate Recall@10;
- seen vs unseen codes;
- candidate-generation vs ranking failures;
- terminology-only vs historical-memory value;
- validation-selected AUTO/HUMAN_REVIEW workload;
- grounded evidence and expert rationale review.

### Audited advanced run

```bash
python -m pip install -e '.[models]'

python scripts/run_cadec_v0013.py \
  --cadec-root /path/to/CADEC \
  --terminology /secure/path/meddra_candidates.csv \
  --output-dir outputs/cadec_v0013 \
  --target-auto-accuracy 0.95 \
  --dense-model /path/or/model-name \
  --cross-encoder-model /path/or/model-name
```

Both advanced model arguments are optional. With neither supplied, the run remains a clean lexical/historical baseline and still uses validation-only model selection.

Main outputs include:

```text
outputs/cadec_v0013/
├── data/
├── audit/
├── benchmark/
│   ├── model_selection.csv
│   ├── model_provenance.json
│   ├── metrics.json
│   ├── results_contract.json
│   ├── frozen_policy.json
│   ├── predictions.csv
│   ├── explanation_quality.json
│   └── explanations.html
├── casebook/
│   ├── review_casebook.csv
│   └── review_casebook.html
└── pipeline_summary.json
```

### Optional frozen-candidate DeepSeek reranking

Fix the prompt/model policy on development or validation first. Then, for an explicitly allowed public/synthetic dataset:

```bash
python scripts/rerank_frozen_candidates_deepseek.py \
  --records outputs/cadec_v0013/data/cadec_split.csv \
  --predictions outputs/cadec_v0013/benchmark/validation_predictions.csv \
  --split val \
  --output-dir outputs/cadec_v0013/deepseek_val \
  --data-classification public \
  --allow-external-llm
```

Do not tune the prompt on held-out TEST labels.

## Benchmark B — MIMIC-IV-Note → ICD-10

Task type: **multi-label document coding**.

The v0.0.12 audited patient-disjoint MIMIC path remains available:

```bash
python scripts/run_mimic_v0012.py \
  --records /secure/derived/mimic_icd10_records.csv \
  --terminology /secure/derived/icd10_terminology.csv \
  --output-dir /secure/results/mimic_v0012 \
  --target-proposal-precision 0.95
```

The explanation quality gate added in v0.0.13 also applies to per-code MIMIC explanations. A target proposal precision applies to **individual code proposals**, not to complete-note automation.

MIMIC source and derived patient-level data must stay outside the public repository.

## Explainability output

For each proposed code, MedCode can show:

```text
Code
MedDRA / ICD-10 code + preferred term

Evidence
exact verbatim source span(s) + character offsets

Why this code?
terminology mapping + optional historical support

Model support
lexical / dense / reranker scores where available

Faithfulness
selected-code score on original input
evidence-only score
evidence-removed score

Clinical context
affirmed / negated / uncertain / family-history / historical

Final operational decision
AUTO_CANDIDATE / CODE_PROPOSAL / HUMAN_REVIEW
```

For CADEC, task-provided BRAT mention spans are preserved exactly rather than re-found by naive string matching.

## Evidence boundary

A result is not automatically reportable because the model is large or the metric looks strong.

The existing Results Contract and dataset-readiness gates still require:

- external reference labels;
- group-disjoint held-out TEST data;
- no TEST-derived terminology leakage;
- no TEST tuning;
- non-synthetic data;
- recorded provenance;
- dataset audit readiness.

An explicit audit override may continue a debugging run, but results are forced to non-reportable status.

## Data governance

Do not commit:

- raw patient/participant-level clinical data;
- MIMIC notes or derived patient-level tables;
- BSRBR free text or restricted cohort extracts;
- governed rationale annotations;
- a full licensed MedDRA distribution;
- API keys or secrets.

`DEEPSEEK_API_KEY` is read from the environment/GitHub secret only when the optional external-LLM path is explicitly invoked.

## Documentation

- `docs/REAL_BENCHMARK_READINESS.md` — pre-flight audit and reportability chain.
- `docs/MODEL_QUALITY_V0013.md` — dense retrieval, reranking, DeepSeek separation, and explanation quality gates.
- `docs/EXPLAINABILITY.md` — rationale/evidence design and methodological caveats.
- `docs/BENCHMARKS.md` — CADEC vs MIMIC task definitions and metric boundaries.

No real CADEC, MIMIC, ICD-10, or MedDRA performance number is implied by the repository code alone. Clinical-performance claims require execution on the corresponding held-out human-reference dataset under the Results Contract.
