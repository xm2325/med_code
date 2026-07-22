# MedCode — v0.0.9

Research prototype for using historical expert-coded clinical free text as a coding memory for future longitudinal cohort follow-up.

## Research question

Can historical expert coding reduce future manual coding workload while keeping agreement with expert reference coding above a prespecified target?

The intended transfer setting is a cohort such as BSRBR-RA: earlier adverse-event/comorbidity free text has expert terminology codes, while later follow-up may require lower-cost coding support.

## v0.0.9

v0.0.9 turns the benchmark from a single-score evaluation into a clinically useful error and workload analysis:

```text
CADEC / other human-labelled free text
        -> document-level TRAIN / VAL / TEST
        -> validation-only history-weight selection
        -> untouched TEST evaluation
        -> seen vs unseen historical codes
        -> candidate-generation vs ranking failures
        -> terminology-only vs historical-memory comparison
        -> validation-selected AUTO / HUMAN_REVIEW policy
        -> workload-reduction stress test at 90/95/98/99% targets
        -> results_contract.json + frozen_policy.json
```

Two methodological corrections are central to this release.

First, the AUTO/REVIEW threshold now **maximises validation coverage subject to the prespecified validation accuracy target**. The previous first-feasible-threshold rule could unnecessarily minimise automation. TEST remains untouched during threshold selection.

Second, terminology retrieval and historical-case retrieval now use separate TF-IDF spaces. Therefore `history_weight=0` is a genuine terminology-only baseline: historical text cannot alter its vocabulary or IDF weights. This makes the estimated value of historical expert coding interpretable as an actual ablation rather than a partially shared representation.

## Quick start

```bash
python -m pip install -e .
pytest -q
python scripts/run_demo.py
```

## Real-data benchmark

```bash
python scripts/prepare_cadec_public.py \
  --cadec-root /path/to/CADEC \
  --output data/processed/cadec.csv

python scripts/split_real_dataset.py \
  --records data/processed/cadec.csv \
  --output data/processed/cadec_with_split.csv

python scripts/run_real_benchmark.py \
  --records data/processed/cadec_with_split.csv \
  --terminology /secure/path/meddra_candidates.csv \
  --output-dir outputs/cadec_v009 \
  --reference-labels-external

python scripts/validate_benchmark.py \
  --benchmark-dir outputs/cadec_v009
```

For later BSRBR work, prefer a temporal split so later expert-coded follow-up remains untouched until final evaluation. A small newly double-coded/adjudicated sample should be used before making prospective workload-reduction claims.

## Main outputs

Primary outputs:

- `metrics.json`
- `predictions.csv`
- `model_selection.csv`
- `results_contract.json`
- `frozen_policy.json`
- `report.html`

v0.0.9 analysis outputs:

- `open_set_metrics.csv` — overall, seen-code, and unseen-code performance
- `candidate_retrieval_diagnostics.csv` — gold candidate rank and per-case failure type
- `failure_summary.csv` — candidate-generation vs ranking failures
- `historical_memory_value.json` — held-out terminology-only vs selected historical-memory comparison
- `coverage_accuracy.csv` and `coverage_accuracy.png` — descriptive held-out coverage/accuracy relationship
- `policy_stress_test.csv` and `policy_workload.png` — validation-selected policies evaluated unchanged on TEST at 90/95/98/99% targets
- `open_set_accuracy.png` — seen-code vs unseen-code exact agreement
- `terminology_only_test_predictions.csv` — pre-specified baseline predictions

See `docs/EVALUATION_OUTPUTS.md` for interpretation rules.

## Evidence boundary

Every benchmark writes a Results Contract. A run is reportable only when it uses an external human reference, a group-disjoint held-out TEST set, no TEST-derived terminology leakage, no TEST tuning, non-synthetic data, and recorded provenance.

Synthetic or oracle runs remain useful for software checks but cannot be presented as medical performance evidence. The TEST coverage-accuracy curve is descriptive and must not be used to select the deployment threshold.

## Data and terminology

Do not commit raw participant-level data, processed sensitive data, or a full licensed MedDRA distribution. Supply terminology locally under the applicable licence.

This repository is a research prototype, not a clinical decision system.
