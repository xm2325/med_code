# Real-data runbook — v0.0.8

## Goal

Use a public human-labelled dataset to test the coding method before BSRBR transfer.

## CADEC path

1. Obtain CADEC from its official source under the applicable data terms.
2. Do not commit the raw corpus to this repository.
3. Parse it:

```bash
python scripts/prepare_cadec_public.py \
  --cadec-root /path/to/CADEC \
  --output data/processed/cadec.csv
```

4. Review the generated `.parse_stats.json` and manually inspect a sample of parsed mention/code pairs before trusting benchmark results.
5. Create a document-disjoint split:

```bash
python scripts/split_real_dataset.py \
  --records data/processed/cadec.csv \
  --output data/processed/cadec_split.csv
```

6. Supply a locally authorised terminology table with `code,term` and optional `synonyms` columns.
7. Run:

```bash
python scripts/run_real_benchmark.py \
  --records data/processed/cadec_split.csv \
  --terminology /secure/path/meddra_candidates.csv \
  --output-dir outputs/cadec_v008 \
  --reference-labels-external
```

8. Validate the result gate:

```bash
python scripts/validate_benchmark.py --benchmark-dir outputs/cadec_v008
```

## Interpretation

Primary coding metrics are Accuracy@1 and Accuracy@5. Also report AUTO candidate rate, AUTO candidate accuracy, and human review rate.

`results_contract.json` controls whether a run is reportable. Synthetic data, oracle terminology built from held-out labels, split leakage, TEST tuning, missing external human reference labels, or missing provenance make a run non-reportable.

## BSRBR transfer

For BSRBR, prefer a prespecified temporal split rather than a random record split. Historical codes are a reference standard and should be audited for coding inconsistency. Before prospective use, create a small new double-coded/adjudicated sample and re-check the frozen policy.
