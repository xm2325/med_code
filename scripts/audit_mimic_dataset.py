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
from cohortcoder.mimic_audit import write_mimic_audit_artifacts


def main() -> None:
    p = argparse.ArgumentParser(description="Audit prepared MIMIC ICD-10 benchmark inputs")
    p.add_argument("--records", required=True)
    p.add_argument("--terminology", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-size", type=int, default=30)
    args = p.parse_args()

    records = pd.read_csv(args.records, dtype=str, keep_default_na=False)
    terminology = load_terminology_knowledge(args.terminology, coding_system="ICD-10")
    audit = write_mimic_audit_artifacts(records, terminology, args.output_dir, sample_size=args.sample_size)
    print(json.dumps(audit, indent=2))
    if not audit["ready_for_benchmark"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
