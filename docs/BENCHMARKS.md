# Dual benchmark design — v0.0.11

v0.0.11 uses two deliberately different benchmark profiles. They share the same principles for evidence grounding and rationale evaluation, but they do **not** share a single coding metric.

## 1. CADEC → MedDRA

**Task:** single-label concept normalization for an annotated adverse-event mention.

```text
patient-authored post
    -> annotated ADE / symptom / disease mention
    -> one MedDRA concept
    -> exact source evidence
    -> why this concept?
```

Primary coding metrics:

- Accuracy@1
- Accuracy@5
- candidate recall@10
- validation-selected AUTO/HUMAN_REVIEW coverage and agreement

Split unit: source document/post. Mentions from one source document must remain in one split.

CADEC is described by Karimi et al. (2015) as a corpus of patient-authored medical forum posts with controlled-vocabulary normalization including MedDRA. Use the source corpus under its applicable CSIRO data licence. Do not redistribute a licensed full MedDRA terminology distribution in this repository.

Reference:

- https://pubmed.ncbi.nlm.nih.gov/25817970/

## 2. MIMIC-IV-Note → ICD-10

**Task:** multi-label document coding.

```text
discharge summary
    -> multiple ICD-10 diagnosis codes
    -> one evidence/rationale object per proposed code
```

A discharge summary can have many diagnosis codes. The pipeline therefore does **not** duplicate one note into multiple fake single-label examples.

Primary coding metrics:

- micro/macro F1 for a validation-selected per-code proposal threshold
- precision@5/10/20
- recall@5/10/20
- seen-code vs unseen-code recall

Split unit: `subject_id`, not note ID. All hospitalizations for one patient remain in one split.

### Data versions

MIMIC-IV-Note v2.2 is the current published note release on PhysioNet and contains discharge summaries that link to MIMIC-IV. For a reproducible default benchmark, pair MIMIC-IV-Note v2.2 with MIMIC-IV v2.2 diagnosis tables unless you intentionally validate another version pairing and record it in the experiment manifest.

MIMIC-IV v3.1 is a newer structured-data release. Do not silently mix versions: record the exact MIMIC-IV-Note and MIMIC-IV versions used.

Official resources:

- MIMIC-IV-Note v2.2: https://physionet.org/content/mimic-iv-note/2.2/
- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/

MIMIC-IV-Note is credentialed-access data. PhysioNet requires credentialing, required training, and the data use agreement. Do not commit notes, patient-level prepared tables, rationale annotations derived from MIMIC, or other restricted derivatives to this public repository.

PhysioNet also states that derived MIMIC datasets/models should be treated as containing sensitive information and shared under the source agreement when appropriate.

## Why the metrics must stay separate

The following comparison is invalid:

```text
CADEC Accuracy@1 = 80%
MIMIC micro-F1 = 55%
therefore CADEC is "better"
```

They are different prediction tasks with different label cardinality and text domains.

`compare_benchmarks.py` therefore creates a side-by-side report but intentionally sets `pooled_score = null`.

## Per-code proposal precision is not full-note automation

For MIMIC, a validation policy may target 95% precision among proposed ICD codes. This means:

> among code proposals made above the frozen threshold, approximately 95% were correct on validation.

It does **not** mean:

> 95% of discharge summaries can be completely automatically coded.

The frozen policy records:

```text
policy_unit = individual_code_proposal
full_note_automation_claim_allowed = false
```

A full-note automation claim would require a separate policy that addresses missing codes as well as false positive codes.

## Rationale / evidence evaluation

Both benchmark profiles can produce:

- verbatim evidence spans with character offsets;
- terminology/knowledge provenance;
- historical expert-coded support;
- retain/remove score diagnostics for faithfulness;
- expert plausibility review templates.

Where human rationale annotations are available, v0.0.11 also evaluates predicted evidence against the reference spans by record and code using character-level precision, recall, and F1.

Expected rationale annotation schema:

```text
record_id,code,start,end,quote
```

Multiple rows may be supplied for multiple rationale spans for the same record-code pair.

## Relationship to Mingyang Li's work

Li et al. (EACL 2026) distinguish rationale **faithfulness** from **plausibility** and construct a multi-granular rationale-annotated ICD-10 dataset based on MIMIC-IV. v0.0.11 follows the same conceptual separation while keeping the implementation application-oriented:

- coding performance remains a separate outcome;
- evidence must be traceable to source text;
- model-centric retain/remove diagnostics test faithfulness;
- expert annotations/review test plausibility;
- optional LLM wording cannot silently change the selected code or invent evidence.

Reference:

- https://aclanthology.org/2026.eacl-long.232/

The rationale dataset described in that work is based on MIMIC-IV and must be handled consistently with its access/governance conditions. This repository does not bundle it or assume it is freely redistributable.
