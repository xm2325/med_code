# Explainable coding design — v0.0.10

## User-facing objective

For each proposed coding label, the system should answer four separate questions:

```text
Clinical record / follow-up text
        ↓
ICD-10 or MedDRA code
        ↓
WHY this code?
        ↓
WHICH exact words in the record support it?
        ↓
WHAT terminology / historical coding evidence supports the mapping?
        ↓
How faithful is the highlighted evidence to the model score?
```

The explanation is downstream of code selection. It does **not** get permission to silently change the code.

## Relationship to Mingyang Li's work

v0.0.10 is informed by two ideas from Mingyang Li's Manchester PhD work and the EACL 2026 paper *Evaluation and LLM-Guided Learning of ICD Coding Rationales*:

1. explanation quality should distinguish **faithfulness** (does the rationale reflect information that affects the model decision?) from **plausibility** (does a clinical expert judge the rationale to be appropriate?);
2. external structured knowledge and LLM-generated rationale labels can support explainable coding, but explanation spans still need systematic evaluation.

Reference: Li et al., EACL 2026, DOI `10.18653/v1/2026.eacl-long.232`.

Manchester thesis: *Explainable ICD Coding with External Knowledge and its Evaluation* (Mingyang Li, PhD, 2026).

This repository is intentionally more application-oriented. The first requirement is an auditable coding-support output that a clinician/coder can inspect record by record.

## Layer 1 — exact evidence spans

`extract_evidence_spans()` returns only text copied verbatim from the supplied record, with stable character offsets:

```json
{
  "start": 31,
  "end": 49,
  "quote": "severe muscle pain",
  "source": "explicit_mention"
}
```

The UI can therefore highlight the original text rather than present a free-form explanation with no traceable source.

## Layer 2 — external terminology knowledge

A terminology CSV can include:

```text
system,code,term,synonyms,definition,hierarchy,knowledge_source
```

The same pipeline supports one coding system per run, for example ICD-10 **or** MedDRA. The system column is retained in every explanation.

`term` remains human-readable. `synonyms` and `definition` are added to a separate `search_text` field for retrieval, so explanations do not display a concatenated pseudo-term.

Do not commit a licensed MedDRA distribution to the repository.

## Layer 3 — historical expert coding memory

Retrieved historical cases are shown as provenance, not as proof that the new record has the same diagnosis. The explanation can state that a similar historical expert-coded expression received the same code, together with similarity, while keeping terminology evidence separate.

## Layer 4 — model-centric faithfulness

Following the retain/remove rationale idea used in explainable ICD coding research, v0.0.10 records three scores for the selected code:

```text
original_code_score

evidence_only_code_score

evidence_removed_code_score
```

Diagnostic quantities:

```text
sufficiency_gap = original_score - evidence_only_score
comprehensiveness_drop = original_score - evidence_removed_score
```

Lower sufficiency gap and higher comprehensiveness drop indicate stronger model-centric support from the extracted evidence.

Important: MedCode currently uses retrieval scores, not calibrated ICD classifier probabilities. These values are therefore **perturbation diagnostics analogous to** sufficiency/comprehensiveness, not directly comparable to published classifier-based values.

## Layer 5 — human-centric plausibility

`rationale_review_template.csv` is generated for expert review with fields such as:

```text
expert_code_supported
expert_evidence_complete
expert_rationale_correct
expert_comments
```

A fluent LLM explanation is not treated as plausible merely because it reads well. Plausibility claims require human/expert reference annotations or structured review.

## Optional DeepSeek rationale generation

DeepSeek is optional and is used as a **locked-code rationale writer**, not as an unrestricted coding agent.

The prompt receives:

- the already-selected code and term;
- exact allowed evidence quotes;
- terminology synonyms/definition/hierarchy;
- optionally, approved TRAIN/development few-shot rationale examples.

It does not receive permission to generate a different code. Post-generation checks reject a response if:

- the code changes;
- an evidence quote is not copied exactly from the allowed evidence set;
- the rationale is missing or malformed.

Rejected output falls back to the deterministic grounded explanation.

### DeepSeek secret

The API client reads only:

```text
DEEPSEEK_API_KEY
```

from the environment. The key is never written to output files or logs.

The current default model is configurable and set to `deepseek-v4-pro` in v0.0.10.

### Data-governance guard

External DeepSeek calls are allowed only when both are true:

1. `--allow-external-llm` is explicitly supplied;
2. `--data-classification` is `public` or `synthetic`.

`restricted` and `private` clinical data are blocked by the client. For BSRBR or other restricted cohorts, use an institutionally approved/local model endpoint unless governance explicitly permits external processing.

By default the external LLM receives grounded evidence spans rather than the full clinical note, reducing unnecessary data disclosure.

## Outputs

`explanations.csv` — one auditable explanation per coding decision.

`explanations.jsonl` — structured evidence, knowledge, historical support and faithfulness fields.

`explanations.html` — record-level review page with highlighted evidence.

`explainability_metrics.json` — grounded/verbatim rates plus aggregate faithfulness diagnostics.

`rationale_review_template.csv` — expert plausibility review template.

## Example interpretation

```text
Proposed code: MedDRA <code> — Myalgia

Evidence:
"severe muscle pain"

Why:
The proposed coding label is grounded in this exact record text, which overlaps
terminology expressions for Myalgia. A similar historical expert-coded example may
be shown separately as provenance.

Faithfulness:
Evidence-only score remains close to the original score;
removing the evidence substantially reduces the selected-code score.
```

This is a coding-support explanation. It is not a new clinical diagnosis and should not add facts that are absent from the source record.
