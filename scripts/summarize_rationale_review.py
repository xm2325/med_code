#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TRUE_VALUES = {"1", "true", "yes", "y", "pass", "supported", "complete", "correct"}
FALSE_VALUES = {"0", "false", "no", "n", "fail", "unsupported", "incomplete", "incorrect"}


def parse_binary(value: object) -> float | None:
    text = str(value).strip().lower()
    if not text:
        return None
    if text in TRUE_VALUES:
        return 1.0
    if text in FALSE_VALUES:
        return 0.0
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed expert plausibility review fields")
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    df = pd.read_csv(args.review_csv).fillna("")
    metrics = {}
    for column in ["expert_code_supported", "expert_evidence_complete", "expert_rationale_correct"]:
        if column not in df:
            continue
        values = [parsed for parsed in (parse_binary(value) for value in df[column]) if parsed is not None]
        metrics[column] = {
            "n_rated": len(values),
            "positive_rate": sum(values) / len(values) if values else None,
        }
    metrics["n_rows"] = len(df)
    text = json.dumps(metrics, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
