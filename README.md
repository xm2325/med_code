# MedCode — v0.3.1

MedCode is an evidence-grounded clinical coding and phenotyping research toolkit. The original coding workflow remains available, while the current research track focuses on **rheumatoid arthritis (RA) comorbidity ascertainment from clinical free text**, with coding treated as a downstream normalization layer rather than the primary scientific task.

## RA comorbidity scientific track

```text
MIPA public-label feasibility
        ↓
authorised local-note phenotyping
        ↓
subject-disjoint Stage 1/2 acceptance
        ↓
Gold × Text × Structured discordance
        ↓
subject-cluster confirmatory inference
        ↓
BSRBR-RA multimorbidity impact
```

Scientific statuses are intentionally separated:

- `PASS` — the claim is supported by the project's required evidence.
- `EXTERNAL_SUPPORT_ONLY` — published literature supports feasibility, but this is not a MedCode performance result.
- `PENDING_REAL_DATA` — authorised notes/predictions or matched structured data are still required.
- `PENDING_BSRBR_RA` — the claim requires representative/longitudinal RA data rather than a phenotype-targeted benchmark.

The overall project remains `NOT_YET_SCIENTIFICALLY_CONFIRMED` until all claim-specific gates are supported. Engineering success, synthetic end-to-end tests, or published results from another pipeline cannot substitute for MedCode's own clinical validation.

Repeated notes from the same patient are not treated as independent observations in confirmatory inference. The v0.3.1 scientific acceptance layer uses **subject-cluster bootstrap resampling** for uncertainty around recovery and sensitivity improvement.

Key RA-track scripts:

- `scripts/run_mipa_public_pilot.py`
- `scripts/generate_mipa_local_predictions.py`
- `scripts/run_mipa_local_phenotyping.py`
- `scripts/run_three_way_discordance.py`
- `scripts/run_scientific_acceptance.py`

Executed public-label outputs are stored under `results/mipa_public_pilot/`. Their RA co-occurrence fractions describe the benchmark sample and must **not** be reported as RA population prevalence estimates.

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
for every displayed candidate:
exact source evidence + terminology support + historical provenance + rationale
              ↓
persistent expert review
              ↓
versioned feedback ledger
              ↓
replayable audit trail
```

A code is never considered explainable merely because an LLM wrote a plausible paragraph. For each candidate MedDRA/ICD code, MedCode keeps source evidence, terminology support, historical provenance, rationale, and faithfulness/context checks separate. Missing evidence is shown as missing; it must not be invented.

The coding release contract and historical documentation remain under `release/` and `docs/`.