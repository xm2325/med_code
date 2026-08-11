# Paid LLM workflow policy

This repository separates ordinary CI from experiments that can spend paid LLM API quota.

## Rule: paid LLM experiments are manual-only

Any workflow that can call `DEEPSEEK_API_KEY` must:

1. use `workflow_dispatch` only; no automatic `push` or `pull_request` trigger;
2. require an explicit `confirm_paid_run=true` input before the paid job can start;
3. check that `DEEPSEEK_API_KEY` is configured before any clinical case is sent;
4. preserve prompts, model configuration, data/version metadata, manifests, and outputs as reproducibility artifacts;
5. never create a new one-off workflow when an existing parameterised experiment workflow can represent the same comparison.

Deleting or rotating the DeepSeek API key is a safe emergency stop for API spending, but it is not the normal scheduling mechanism.

## Active CodiEsp experiment workflow

### `codiesp-icd-kb-ab-50.yml`

Primary paired experiment:

- Direct DeepSeek Pro/max-thinking: full clinical text only.
- ICD knowledge-assisted DeepSeek Pro/max-thinking: same clinical text plus dynamically retrieved official ICD structured knowledge.
- Same fixed cases and evaluation schema for both arms.
- Manual paid-run confirmation required.

This workflow is the only active paid CodiEsp A/B workflow. Future 250-case confirmation should extend/parameterise this workflow rather than create another `*-250.yml` workflow.

## Manual legacy/reproducibility workflows

These are retained only for explicit historical reproduction and are manual-only:

- `codiesp-icd10-benchmark.yml` — legacy TF-IDF / candidate-assisted benchmark.
- `public-interim-benchmarks.yml` — public interim dataset audits/proxy experiments.
- `waiting-mimic-experiments.yml` — historical combined NHSE/Synthea/CodiEsp waiting-MIMIC experiments.

## Removed redundant workflow triggers

The following one-off CodiEsp workflows were removed because their scripts/results are historical and their functionality is superseded by the paired A/B workflow:

- `codiesp-direct-thinking-50.yml`
- `codiesp-direct-flash-50.yml`
- `codiesp-recall-first-50.yml`
- `codiesp-direct-deepseek-thinking.yml`
- `codiesp-dashboard-data.yml`

Their historical scripts, committed result summaries, Git history, and prior Actions artifacts are intentionally not deleted.

## Workflows that may remain automatic

Non-paid engineering/scientific validation workflows may run on normal code changes, including CI and MIPA public/synthetic acceptance checks, provided they do not call a paid external LLM API.
