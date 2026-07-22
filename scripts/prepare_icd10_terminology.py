#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.icd import prepare_icd10_terminology_from_mimic_dictionary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ICD-10 terminology candidates from MIMIC-IV d_icd_diagnoses")
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = prepare_icd10_terminology_from_mimic_dictionary(args.dictionary, args.output)
    print(f"Prepared {len(frame)} ICD-10 terminology rows at {args.output}")


if __name__ == "__main__":
    main()
