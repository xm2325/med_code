# Changelog

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
