# MedCode

Research prototype for historical expert-assisted clinical coding of longitudinal cohort free text.

## Goal

MedCode tests whether historical human coding can support future clinical-event and comorbidity coding while routing uncertain cases to expert review. The target transfer setting is a longitudinal cohort such as BSRBR-RA, where earlier free-text adverse-event/comorbidity records have expert terminology labels and later records may require lower-cost coding support.

## v0.0.6

The current release provides:

- document/group-disjoint train, validation and test evaluation;
- historical coding-memory retrieval and terminology retrieval;
- lexical and optional dense retrieval backends;
- optional second-stage reranking interfaces;
- seen-code and unseen-code/open-set analysis;
- validation-only model selection;
- separate model-selection, confidence-calibration and policy-validation roles when data permit;
- frozen confidence calibration and AUTO/REVIEW thresholds;
- a single `frozen_policy.json` for reproducible inference;
- candidate-recall and ranking-error diagnostics;
- bootstrap confidence intervals at source-document level;
- publication/deployment guards that distinguish synthetic smoke tests from real human-reference evaluation;
- CADEC and ALTA data adapters;
- a BSRBR-style uncoded inference example;
- CI tests and reproducible command-line entry points.

## Core study design

```text
Historical expert-coded records
        |
        +--> historical coding memory
        |
Official terminology --> candidate retrieval
        |
        v
candidate ranking / optional reranking
        |
        v
confidence calibration
        |
        +--> AUTO candidate
        |
        +--> HUMAN REVIEW
```

The main research questions are not only whether a model can predict a code, but whether historical expert coding adds value beyond terminology retrieval, how performance changes for unseen codes, and how much manual review can be reduced at a prespecified coding-agreement target.

## Quick start

```bash
python -m pip install -r requirements.txt
pytest -q
python scripts/run_demo.py
```

For a real benchmark, use externally supplied human reference labels and an authorised terminology resource. Synthetic demo metrics are software checks only and must not be reported as medical performance.

## Real-data path

```text
CADEC human-labelled data
        -> public method development
ALTA MedDRA normalization benchmark
        -> harder external/open-set evaluation
BSRBR historical expert-coded data
        -> temporal retrospective validation
small newly double-coded BSRBR sample
        -> prospective validation
```

See `docs/REAL_DATA_RUNBOOK.md`, `docs/BSRBR_TRANSFER_PROTOCOL.md`, and `docs/RERANKING_PROTOCOL.md` for the intended evaluation workflow.

## Data and terminology

Raw participant-level datasets, processed sensitive data, model outputs, environment files, and licensed terminology distributions should not be committed to this repository. MedDRA terminology files must be supplied separately under the applicable licence.

## Status

v0.0.6 is a research prototype. The bundled synthetic data are for smoke testing only. Claims about clinical coding accuracy require evaluation against real held-out human reference labels.
