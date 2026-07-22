#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.mednorm import assign_cross_dataset_split, build_train_derived_terminology, fetch_hf_mirror_rows, mednorm_data_card, prepare_mednorm_single_meddra
from cohortcoder.realdata import run_real_benchmark


def main() -> None:
    p = argparse.ArgumentParser(description="Run a real MedNorm MedDRA concept-normalisation benchmark")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--test-source", default="CADEC")
    p.add_argument("--external-terminology", help="Authorised MedDRA terminology CSV; enables full candidate-space experiment")
    p.add_argument("--target-auto-accuracy", type=float, default=0.95)
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = fetch_hf_mirror_rows(max_rows=a.max_rows)
    records = assign_cross_dataset_split(prepare_mednorm_single_meddra(raw), test_source=a.test_source)
    train = records[records.split == "train"].copy()
    if a.external_terminology:
        import pandas as pd
        terminology = pd.read_csv(a.external_terminology, dtype=str, keep_default_na=False)
        candidate_source = "external_authorised_terminology"
    else:
        terminology = build_train_derived_terminology(train)
        candidate_source = "train_derived_closed_code"

    records.to_csv(out / "mednorm_records.csv", index=False)
    terminology.to_csv(out / "candidate_terminology.csv", index=False)
    (out / "data_card.json").write_text(json.dumps(mednorm_data_card(), indent=2), encoding="utf-8")

    metrics = run_real_benchmark(
        records,
        terminology,
        out / "benchmark",
        target_auto_accuracy=a.target_auto_accuracy,
        external_human_reference=True,
        data_is_synthetic=False,
    )
    metrics["candidate_space"] = candidate_source
    metrics["important_interpretation"] = (
        "Without --external-terminology this is an honest TRAIN-derived closed-code diagnostic: unseen TEST codes are structurally unavailable. "
        "Do not describe it as a full open-set MedDRA benchmark."
    )
    (out / "real_data_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
