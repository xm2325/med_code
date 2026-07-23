#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.ra_comorbidity import (
    confidence_review_curve,
    evaluate_discordance,
    public_mipa_ra_summary,
    validate_discordance_table,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Run RA hidden-comorbidity feasibility/discordance analyses")
    p.add_argument("--mipa-labels", help="Public MIPA labels.csv for Stage A feasibility")
    p.add_argument("--discordance-table", help="Long-format Gold x Code x Text CSV for Stage B")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    if not args.mipa_labels and not args.discordance_table:
        raise SystemExit("Provide --mipa-labels and/or --discordance-table")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.mipa_labels:
        labels = pd.read_csv(args.mipa_labels)
        summary, counts = public_mipa_ra_summary(labels)
        (output / "public_mipa_ra_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        counts.to_csv(output / "public_mipa_ra_patient_comorbidity_counts.csv")

    if args.discordance_table:
        table = pd.read_csv(args.discordance_table)
        validate_discordance_table(table, require_evidence=True)
        metrics, patterns = evaluate_discordance(table)
        metrics.to_csv(output / "ra_discordance_metrics.csv", index=False)
        patterns.to_csv(output / "ra_discordance_patterns.csv", index=False)
        curve = confidence_review_curve(table)
        if not curve.empty:
            curve.to_csv(output / "ra_review_workload_curve.csv", index=False)

    print(json.dumps({"status": "completed", "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
