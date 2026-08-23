# Private ADEcoding PT feasibility experiment

This repository publishes the experiment code and prompts, but deliberately does not distribute the source archive, extracted records, derived input CSVs, model-result CSVs, checkpoints, or the MedDRA Coding SOP PDF.

The source data used in the feasibility run contained legacy PT names but no versioned MedDRA PT numeric codes. The experiment therefore emits PT-term candidates only, keeps `selected_pt_codes_json` empty, and records the terminology version as `UNKNOWN`. It must not be described as a MedDRA-version-specific coding result or as clinical accuracy.

## Three independently executed methods

The complete executable prompts are versioned in `scripts/run_adecoding_pt_experiment.py`.

1. `prompt_only` gives Qwen the closed legacy PT-term allowlist. The model may select only supplied `legacy_term_id` values and must copy exact evidence quotes.
2. `extract_lexical` uses Qwen only for exact event-span and assertion extraction. Code candidates come from deterministic BM25, character-trigram, and exact-match retrieval.
3. `hybrid_rag` performs a separate extraction, retrieves Top-K candidates, freezes their IDs and asks Qwen to rank every candidate exactly once. Server validation rejects any addition, removal, duplication, or replacement.

All prompts treat clinical text as untrusted data and prohibit inference of diagnosis, causality, severity, treatment, or unreported facts. Evidence offsets are accepted only when the quote occurs exactly once and matches the original string exactly.

## Private input preparation

Run this only in an approved private environment. The output contains source text and is classified `restricted`.

```bash
python scripts/prepare_adecoding_pt_inputs.py \
  --zip /private/path/to/source.zip \
  --output-dir /private/path/to/experiment/input
```

The source archive is never uploaded by this command. It reads the legacy `sample_errors.csv` member locally and writes a mode-600 derived input CSV plus a manifest.

## Roihu execution

Submit from the repository root. Keep `EXPERIMENT_DIR` on private project scratch and provide the CSC allocation through an environment variable rather than committing it.

```bash
export ROIHU_PROJECT=project_xxxxxxx
export EXPERIMENT_DIR=/scratch/${ROIHU_PROJECT}/${USER}/private/adecoding-pt
export EXPERIMENT_INPUT_FILE=${EXPERIMENT_DIR}/input/adecoding_pt_all_inputs.csv
export EXPERIMENT_ID=all_rows_v1
export EXPERIMENT_LIMIT=5921
export EXPERIMENT_WORKERS=16
export EXPERIMENT_OUTPUT_PREFIX=adecoding_pt_all
sbatch slurm/run_adecoding_pt_experiment.sbatch
```

The Slurm template serves the revision-pinned Qwen model on `127.0.0.1`, disables request access logs, runs offline from the project cache and writes mode-600 checkpoints/results. Adjust resources and concurrency only after a private smoke test.

## Output interpretation

- `LIVE` means the method executed and its payload passed deterministic validation.
- `VALIDATION_FAILED` is a fail-closed evidence/schema/candidate-set rejection, not a system failure.
- `ERROR` is a technical method failure.
- Retrieval scores are relevance scores, not probabilities.
- `AUTO_CANDIDATE` is a workflow route, not a statement of correctness.
- Agreement with legacy targets is fixture agreement, not clinical accuracy.

Duplicate exact inputs may reuse one inference result, while each source row retains its own legacy target and agreement metrics. Conflicting duplicate targets remain visible for review.

For a full 5,921-row verification bundle containing the three generated CSVs,
the derived input CSV, the legacy PT vocabulary and `run_manifest.json`, run:

```bash
python scripts/verify_adecoding_pt_outputs.py /private/path/to/verification-bundle
```

The verifier checks counts, hashes, exact evidence offsets, closed-set IDs,
frozen RAG candidates, wide/long-table agreement, summary recomputation and the
absence of PDF/ZIP files. The verification bundle remains private and ignored.

## Pre-publication safety check

Before every commit, verify that no source data or generated output is staged:

```bash
git status --short
git diff --cached --name-only
git grep -n -E 'project_[0-9]+|/Users/|sample_errors.csv' -- ':!docs/ADECODING_PT_EXPERIMENT.md'
```

Never use `git add -A` in a working tree that contains the private source archive or SOP PDF.
