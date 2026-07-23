#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cohortcoder.discordance import evaluate_three_way_discordance, write_discordance_outputs
from cohortcoder.mipa_phenotyping import DEFAULT_PHENOTYPES


def _local_path(value: str) -> str:
    if value.lower().startswith(("http://", "https://", "s3://", "gs://")):
        raise argparse.ArgumentTypeError("Use local filesystem paths for clinical-data experiments.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate clinician gold × text model × structured-record phenotype discordance."
    )
    parser.add_argument("--labels", required=True, type=_local_path)
    parser.add_argument("--text-predictions", required=True, type=_local_path)
    parser.add_argument("--structured-status", required=True, type=_local_path)
    parser.add_argument("--output-dir", required=True, type=_local_path)
    parser.add_argument("--phenotypes", nargs="+", default=list(DEFAULT_PHENOTYPES))
    parser.add_argument("--upstream-summary", type=_local_path)
    parser.add_argument(
        "--structured-scope-validated",
        action="store_true",
        help="Confirm that the structured-code phenotype definition has been reviewed to match the gold phenotype target.",
    )
    parser.add_argument(
        "--require-confirmatory-eligible",
        action="store_true",
        help="Exit non-zero unless upstream PASS, structured scope validation, and complete coverage are present.",
    )
    args = parser.parse_args()

    summary, metrics, cells = evaluate_three_way_discordance(
        labels_path=args.labels,
        text_predictions_path=args.text_predictions,
        structured_status_path=args.structured_status,
        phenotypes=args.phenotypes,
        upstream_summary_path=args.upstream_summary,
        structured_scope_validated=args.structured_scope_validated,
    )
    write_discordance_outputs(args.output_dir, summary, metrics, cells)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nOutputs: {Path(args.output_dir).resolve()}")

    if args.require_confirmatory_eligible and not summary["interpretation"]["confirmatory_underrecording_language_allowed"]:
        raise SystemExit("Three-way analysis is not confirmatory-eligible; see summary.json")


if __name__ == "__main__":
    main()
