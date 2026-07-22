#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.rationale_metrics import evaluate_rationale_overlap, validate_rationale_offsets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted evidence spans against human rationale annotations")
    parser.add_argument("--explanations", required=True, help="explanations.csv from MedCode")
    parser.add_argument("--reference", required=True, help="CSV: record_id,code,start,end[,quote]")
    parser.add_argument("--records", required=True, help="Source records CSV containing record_id,text")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predicted = pd.read_csv(args.explanations, dtype=str, keep_default_na=False)
    reference = pd.read_csv(args.reference, dtype=str, keep_default_na=False)
    records = pd.read_csv(args.records, dtype=str, keep_default_na=False)

    audit = validate_rationale_offsets(reference, records)
    if not audit["valid"]:
        raise ValueError(f"Reference rationale offset audit failed: {audit}")
    summary, detail = evaluate_rationale_overlap(predicted, reference)
    (output / "rationale_reference_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "rationale_overlap_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    detail.to_csv(output / "rationale_overlap_detail.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
