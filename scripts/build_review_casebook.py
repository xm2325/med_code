#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.review_casebook import build_review_casebook, write_review_casebook


def _attach_source_context(predictions: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    source = records.copy().fillna("")
    if "split" in source.columns and (source["split"].astype(str) == "test").any():
        source = source[source["split"].astype(str) == "test"].copy()
    source["_occurrence"] = source.groupby("record_id").cumcount()
    pred = predictions.copy().fillna("")
    pred["_occurrence"] = pred.groupby("record_id").cumcount()
    columns = ["record_id", "_occurrence", "text"]
    if "mention" in source.columns:
        columns.append("mention")
    merged = pred.merge(source[columns], on=["record_id", "_occurrence"], how="left", validate="one_to_one")
    return merged.drop(columns=["_occurrence"])


def main() -> None:
    p = argparse.ArgumentParser(description="Build a priority clinical/coder review casebook")
    p.add_argument("--predictions", required=True)
    p.add_argument("--explanations")
    p.add_argument("--records", help="Optional source records used to restore original TEST context")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-cases", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    predictions = pd.read_csv(args.predictions).fillna("")
    if args.records:
        predictions = _attach_source_context(predictions, pd.read_csv(args.records).fillna(""))
    explanations = pd.read_csv(args.explanations).fillna("") if args.explanations else None
    casebook = build_review_casebook(
        predictions,
        explanations,
        max_cases=args.max_cases,
        seed=args.seed,
    )
    write_review_casebook(args.output_dir, casebook)
    print(f"Wrote {len(casebook):,} review cases -> {args.output_dir}")


if __name__ == "__main__":
    main()
