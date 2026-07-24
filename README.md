# MedCode — v0.3.2

MedCode is an evidence-grounded clinical coding and phenotyping research toolkit. The original coding workflow remains available, while the current research track focuses on **rheumatoid arthritis (RA) comorbidity ascertainment from clinical free text**, with coding treated as a downstream normalization layer rather than the primary scientific task.

## RA comorbidity scientific track

```text
MIPA public-label feasibility
        ↓
dual-protocol audit
   ├── protocol-matched benchmark comparison
   └── subject-disjoint confirmatory RA analysis
        ↓
authorised local-note phenotyping
        ↓
Stage 1/2 performance + evidence acceptance
        ↓
Gold × Text × Structured discordance
        ↓
subject-cluster confirmatory recovery inference
        ↓
BSRBR-RA multimorbidity impact assessment
```

### Two evaluation protocols must stay separate

The public MIPA implementation creates phenotype-specific validation/test sets at the **admission (`hadm_id`) level**. Our executed audit reproduces that logic and finds no admission overlap but substantial `subject_id` overlap between validation and test. These protocol-matched results are useful for reproducing/comparing with MIPA studies, but admission-disjointness must not be described as patient-disjointness.

For the primary RA scientific claim, MedCode uses a separate **subject-disjoint** protocol: every note from the same patient remains in one split, and confirmatory uncertainty is estimated by subject-cluster bootstrap resampling.

Metrics from the two protocols must not be mixed. A published MIPA macro-F1 is treated as external feasibility context unless model, phenotype definition, evaluation cohort, and evaluation protocol match.

## Scientific decision states

Scientific acceptance distinguishes evidence from direction of effect:

- `PASS` — the specified directional claim is supported by the project's required evidence.
- `EXTERNAL_SUPPORT_ONLY` — published literature supports feasibility; this is non-gating and is not MedCode performance.
- `PENDING_REAL_DATA` / `PENDING_BSRBR_RA` — required empirical data have not yet been run.
- `NOT_SUPPORTED` — a completed, valid experiment does not support the directional claim.
- `SCIENTIFIC_NO_GO_STAGE12` — the phenotype method failed its prespecified clinical gate; stop or redesign before under-recording claims.
- `SCIENTIFIC_CONCLUSION_DETERMINED_NO_RECOVERY_SUPPORT` — confirmatory analysis was completed but did not support recoverable under-recording.
- `GO_BSRBR_RA` — recovery evidence passed and downstream RA impact analysis is the next gate.
- `SCIENTIFIC_PROGRAM_COMPLETE` — the prespecified BSRBR-RA impact analysis is complete; a material, small, or null downstream effect is scientifically valid.

Engineering success, synthetic end-to-end tests, or published results from another pipeline cannot substitute for MedCode's own authorised clinical validation.

Key RA-track scripts:

- `scripts/run_mipa_public_pilot.py`
- `scripts/run_mipa_split_protocol_audit.py`
- `scripts/generate_mipa_local_predictions.py`
- `scripts/run_mipa_local_phenotyping.py`
- `scripts/run_three_way_discordance.py`
- `scripts/run_scientific_acceptance.py`

Executed public-label and method-audit outputs are stored under `results/mipa_public_pilot/`, `results/mipa_split_protocol_audit/`, and `results/scientific_acceptance/`. Benchmark-sample RA co-occurrence fractions must **not** be reported as RA population prevalence estimates.

## Data governance

Restricted MIPA/MIMIC notes must remain inside an approved compute boundary. The local inference harness itself makes no HTTP/API calls and passes note text only to a user-supplied local child process over stdin. The approved host/container remains responsible for enforcing OS-level network isolation for the child model process.

## Original explainable coding workflow

```text
clinical text
    ↓
terminology retrieval + historical expert-coding memory
    ↓
optional biomedical dense retrieval / reranking
    ↓
ranked code candidates
    ↓
uncertainty + OOD + evidence/context checks
    ├── AUTO_CANDIDATE
    ├── TOP_K_HUMAN_CHOICE
    └── FULL_EXPERT_REVIEW
              ↓
exact source evidence + terminology support + historical provenance + rationale
              ↓
persistent expert review → versioned feedback → replayable audit trail
```

A code or phenotype is never considered explainable merely because an LLM wrote a plausible paragraph. MedCode keeps source evidence, terminology support, historical provenance, rationale, and context/faithfulness checks separate. Missing evidence is shown as missing; it must not be invented.

Historical coding release documentation remains under `release/` and `docs/`.