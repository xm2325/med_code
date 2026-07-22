#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.benchmark_compare import write_dual_benchmark_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CADEC-MedDRA and MIMIC-ICD10 benchmarks without pooling incompatible metrics")
    parser.add_argument("--cadec-dir", required=True)
    parser.add_argument("--mimic-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = write_dual_benchmark_summary(args.cadec_dir, args.mimic_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
