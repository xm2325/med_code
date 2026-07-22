#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.knowledge import load_terminology_knowledge
from cohortcoder.mimic_benchmark import run_mimic_icd10_benchmark
from cohortcoder.mimic_report import write_mimic_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patient-disjoint MIMIC-IV-Note ICD-10 multi-label benchmark")
    parser.add_argument("--records", required=True)
    parser.add_argument("--terminology", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-proposal-precision", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--source-version", default="MIMIC-IV-Note 2.2 + MIMIC-IV 2.2")
    parser.add_argument("--data-is-synthetic", action="store_true")
    parser.add_argument("--reference-labels-external", action="store_true")
    args = parser.parse_args()

    records = pd.read_csv(args.records, dtype=str, keep_default_na=False)
    terminology = load_terminology_knowledge(args.terminology, coding_system="ICD-10")
    metrics = run_mimic_icd10_benchmark(
        records,
        terminology,
        args.output_dir,
        target_proposal_precision=args.target_proposal_precision,
        external_human_reference=args.reference_labels_external,
        data_is_synthetic=args.data_is_synthetic,
        source_version=args.source_version,
        batch_size=args.batch_size,
    )
    write_mimic_report(args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
