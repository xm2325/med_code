# Changelog

## v0.0.14

- Adds explicit uncertainty features from candidate score margin, normalised entropy, and top score.
- Adds three-way routing: `AUTO_CANDIDATE`, `TOP_K_HUMAN_CHOICE`, and `FULL_EXPERT_REVIEW`.
- Adds a conservative OOD hook and keeps explanation/OOD failures downgrade-only.
- Adds Top-K review packets where every displayed candidate has its own exact source evidence, terminology support, historical provenance, and candidate-specific rationale.
- Adds CSV/JSONL/HTML review artifacts so uncertain cases can be resolved by a human without hiding alternative candidates.

## v0.0.13

- Adds optional biomedical dense retrieval through a user-supplied sentence-transformer-compatible model or local path.
- Adds optional cross-encoder reranking over a frozen candidate pool; rerankers cannot create codes outside the supplied terminology dictionary.
- Adds staged validation-only selection of historical-memory, dense-fusion, and cross-encoder weights, with ties resolved in favour of simpler models.
- Adds an advanced CADEC runner that preserves the v0.0.12 audit/readiness gates and freezes the selected model family in `frozen_policy.json`.
- Adds constrained DeepSeek frozen-candidate reranking; responses are rejected when codes are added, removed, or duplicated.
- Keeps DeepSeek candidate reranking separate from locked-code rationale writing so coding and explanation effects can be evaluated independently.
- Adds a conservative explanation quality gate covering verbatim evidence, clinical-context review flags, terminology support, and faithfulness diagnostics.
- Explanation-quality failures may downgrade AUTO/CODE_PROPOSAL decisions to HUMAN_REVIEW but can never promote a review case to automatic handling.
- Adds model provenance, explanation-quality artifacts, advanced-model tests, and frozen-policy reconstruction for explainability.

## v0.0.12

- Adds a stricter CADEC BRAT/MedDRA parser that preserves discontinuous spans instead of collapsing them into one evidence interval.
- Adds source-offset validation and parser statistics so annotation text must agree with the original source characters before results are trusted.
- Adds CADEC pre-flight readiness checks for offset integrity, duplicate/multi-code anomalies, and external terminology coverage.
- Preserves exact CADEC task-input spans in explanation output, avoiding ambiguous first-string-match evidence when the same phrase occurs multiple times.
- Adds a MIMIC pre-flight audit for patient leakage, empty text/labels, ICD-10 terminology coverage, note-length distribution, and codes-per-note distribution.
- Adds one-command audited runners for CADEC->MedDRA and MIMIC-IV-Note->ICD-10.
- Adds priority clinical review casebooks that surface AUTO errors, high-confidence errors, unseen-code errors, evidence failures, and correct controls with original source context.
- Adds release/version stamping to frozen policy and experiment manifests so generated artifacts identify the code release that produced them.
- Keeps DeepSeek optional and downstream of grounded affirmative evidence; no affirmative evidence means no LLM call.

## v0.0.11

- Adds explicit benchmark profiles so CADEC MedDRA normalization and MIMIC ICD-10 document coding are not treated as the same prediction task.
- Adds MIMIC-IV-Note discharge-summary + `diagnoses_icd` preparation with ICD-10 filtering and deterministic patient-level (`subject_id`) splitting.
- Adds a true multi-label historical/terminology retrieval baseline for ICD-10 coding rather than duplicating one discharge summary into fake single-label examples.
- Adds validation-only model selection and a per-code proposal threshold that maximises recall subject to a prespecified proposal precision target.
- Explicitly prevents interpreting per-code proposal precision as the percentage of whole discharge summaries that can be fully auto-coded.
- Adds seen/unseen ICD-code recall, precision/recall@k, micro/macro F1, terminology-only ablation, and proposal-policy stress tests.
- Adds one explanation object per proposed ICD code, with verbatim evidence, terminology provenance, historical support, context audit, and retain/remove faithfulness diagnostics.
- Adds generic human rationale span evaluation by record-code pair using audited character offsets and character-level precision/recall/F1.
- Adds a side-by-side CADEC/MIMIC benchmark report that intentionally does not compute a pooled score across incompatible tasks.
- Adds benchmark/data-governance documentation for credentialed MIMIC-IV-Note and source-version tracking.

## v0.0.10

- Adds exact character-offset evidence spans for each proposed coding label.
- Adds record-level `why this code?` explanations grounded in verbatim source text.
- Adds external terminology knowledge fields for ICD/MedDRA runs: synonyms, definition, hierarchy, and knowledge source.
- Keeps one coding system per benchmark run and preserves clean human-readable terminology labels.
- Adds score-level retain/remove evidence diagnostics analogous to rationale sufficiency and comprehensiveness.
- Adds `explanations.csv`, `explanations.jsonl`, `explanations.html`, `explainability_metrics.json`, and an expert plausibility-review template.
- Adds optional DeepSeek rationale generation using `DEEPSEEK_API_KEY`, with the selected code locked and evidence quotes post-validated.
- Blocks external LLM calls for restricted/private data and requires explicit opt-in for public/synthetic data.
- Adds an end-to-end new-record path: prediction -> code -> evidence -> rationale -> AUTO/REVIEW decision.
- Adds a manual, synthetic-only DeepSeek GitHub Actions smoke workflow so repository secrets are never printed or committed.

## v0.0.9

- Fixes AUTO/REVIEW threshold selection to maximise validation coverage subject to the prespecified accuracy target.
- Separates terminology and historical TF-IDF spaces so `history_weight=0` is a genuine terminology-only baseline whose vocabulary/IDF is not influenced by historical text.
- Adds seen-code vs unseen-code held-out analysis.
- Separates candidate-generation failures from ranking/disambiguation failures.
- Adds a pre-specified terminology-only TEST baseline and historical-memory value analysis.
- Adds descriptive coverage-accuracy output and validation-selected policy stress tests at 90%, 95%, 98%, and 99% targets.
- Adds presentation-ready coverage/accuracy, open-set, and policy-workload figures.
- Expands the HTML report and per-case diagnostics without using TEST for model or threshold selection.

## v0.0.8

- Adds a self-contained real-data pipeline for CADEC/BSRBR-style terminology normalization.
- Adds CADEC BRAT-to-table parsing.
- Adds stable document-level splitting and leakage checks.
- Adds validation-only selection of historical-memory weight.
- Adds frozen AUTO/REVIEW policy export and uncoded-record inference.
- Integrates Results Contract generation into benchmark execution.
- Gates reportability on external human reference labels, held-out test integrity, no TEST-derived terminology leakage, no TEST tuning, non-synthetic data, and recorded provenance.
- Adds benchmark validation CLI and auditable result artifacts.

## v0.0.7

- Added the Results Contract abstraction and conservative reportability defaults.

## v0.0.6

- Added GitHub CI, research-safe ignore rules, core retrieval baseline, BSRBR transfer protocol, and reproducible project structure.
