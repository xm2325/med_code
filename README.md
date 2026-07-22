# MedCode — v0.0.8

Research prototype for using historical expert-coded clinical free text as a coding memory for future longitudinal cohort follow-up.

## Research question

Can historical expert coding reduce future manual coding workload while keeping agreement with expert reference coding above a prespecified target?

The intended transfer setting is a cohort such as BSRBR-RA: earlier adverse-event/comorbidity free text has expert terminology codes, while later follow-up may require lower-cost coding support.

## v0.0.8

v0.0.8 adds a self-contained real-data path:

```text
CADEC/other labelled free text
        -> parser / normalized CSV
        -> document-level train/validation/test
        -> validation-only history-weight selection
        -> untouched TEST evaluation
        -> confidence-based AUTO vs HUMAN_REVIEW
        -> results_contract.json
        -> frozen_policy.json
        -> inference on new uncoded records
```

Every benchmark writes a Results Contract. A run is reportable only when it uses an external human reference, a group-disjoint held-out test set, no TEST-derived terminology leakage, no TEST tuning, non-synthetic data, and recorded provenance.

Synthetic or oracle runs remain useful for software checks but cannot be presented as medical performance evidence.

## Quick start

```bash
python -m pip install -r requirements.txt
pytest -q
python scripts/run_demo.py
```

### Real-data benchmark

```bash
python scripts/prepare_cadec_public.py \
  --cadec-root /path/to/CADEC \
  --output data/processed/cadec.csv

python scripts/run_real_benchmark.py \
  --records data/processed/cadec_with_split.csv \
  --terminology /secure/path/meddra_candidates.csv \
  --output-dir outputs/cadec_v008 \
  --reference-labels-external

python scripts/validate_benchmark.py \
  --benchmark-dir outputs/cadec_v008
```

For later BSRBR work, prefer a temporal split so later expert-coded follow-up remains untouched until final evaluation. A small newly double-coded/adjudicated sample should be used before making prospective workload-reduction claims.

## Main outputs

`metrics.json`, `predictions.csv`, `model_selection.csv`, `error_analysis.csv`, `results_contract.json`, `leakage_audit.json`, `experiment_manifest.json`, `data_fingerprints.json`, `frozen_policy.json`, and `report.html`.

## Data and terminology

Do not commit raw participant-level data, processed sensitive data, or a full licensed MedDRA distribution. Supply terminology locally under the applicable licence.

This repository is a research prototype, not a clinical decision system.