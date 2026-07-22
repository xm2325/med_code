#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.mimic import prepare_mimic_iv_icd10


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a patient-disjoint MIMIC-IV-Note discharge-summary ICD-10 benchmark table"
    )
    parser.add_argument("--discharge", required=True, help="MIMIC-IV-Note discharge.csv or discharge.csv.gz")
    parser.add_argument("--diagnoses", required=True, help="MIMIC-IV diagnoses_icd.csv or .csv.gz")
    parser.add_argument("--dictionary", required=True, help="MIMIC-IV d_icd_diagnoses.csv or .csv.gz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-code-frequency", type=int, default=1)
    parser.add_argument("--max-notes", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = prepare_mimic_iv_icd10(
        args.discharge,
        args.diagnoses,
        args.dictionary,
        args.output,
        min_code_frequency=args.min_code_frequency,
        max_notes=args.max_notes,
        seed=args.seed,
    )
    print(f"Prepared {len(frame)} hospitalization-level records at {args.output}")


if __name__ == "__main__":
    main()
