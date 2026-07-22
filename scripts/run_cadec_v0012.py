#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.analysis import write_evaluation_plots
from cohortcoder.artifact_metadata import stamp_benchmark_artifacts
from cohortcoder.cadec import parse_cadec, write_cadec_audit_artifacts
from cohortcoder.cadec_report import write_cadec_report
from cohortcoder.knowledge import load_terminology_knowledge
from cohortcoder.realdata import assign_document_splits, run_real_benchmark


def main() -> None:
    p = argparse.ArgumentParser(description="Run audited CADEC -> MedDRA benchmark end to end")
    p.add_argument("--cadec-root", required=True)
    p.add_argument("--terminology", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-auto-accuracy", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audit-sample-size", type=int, default=50)
    p.add_argument("--casebook-size", type=int, default=40)
    p.add_argument("--allow-audit-failures", action="store_true")
    args = p.parse_args()

    output = Path(args.output_dir)
    data_dir = output / "data"
    audit_dir = output / "audit"
    benchmark_dir = output / "benchmark"
    casebook_dir = output / "casebook"
    for path in [data_dir, audit_dir, benchmark_dir, casebook_dir]:
        path.mkdir(parents=True, exist_ok=True)

    parsed_path = data_dir / "cadec_parsed.csv"
    split_path = data_dir / "cadec_split.csv"
    records, parse_stats = parse_cadec(args.cadec_root, parsed_path)
    terminology = load_terminology_knowledge(args.terminology, coding_system="MedDRA")
    audit = write_cadec_audit_artifacts(records, audit_dir, terminology, sample_size=args.audit_sample_size)
    if not audit["ready_for_benchmark"] and not args.allow_audit_failures:
        (output / "pipeline_summary.json").write_text(json.dumps({
            "status": "blocked_by_dataset_audit",
            "parse_stats": parse_stats,
            "dataset_audit": audit,
        }, indent=2), encoding="utf-8")
        raise SystemExit("CADEC audit failed. Review audit/dataset_audit.json before benchmarking.")

    split_records = assign_document_splits(records, seed=args.seed)
    split_records.to_csv(split_path, index=False)
    metrics = run_real_benchmark(
        split_records,
        terminology,
        benchmark_dir,
        target_auto_accuracy=args.target_auto_accuracy,
        external_human_reference=True,
        data_is_synthetic=False,
        seed=args.seed,
    )
    stamp_benchmark_artifacts(benchmark_dir, version="0.0.12", benchmark_profile="cadec_meddra_normalization")

    import pandas as pd
    write_evaluation_plots(
        benchmark_dir,
        pd.read_csv(benchmark_dir / "open_set_metrics.csv"),
        pd.read_csv(benchmark_dir / "coverage_accuracy.csv"),
        pd.read_csv(benchmark_dir / "policy_stress_test.csv"),
    )

    subprocess.run([
        sys.executable,
        str(ROOT / "scripts" / "explain_benchmark.py"),
        "--records", str(split_path),
        "--terminology", str(args.terminology),
        "--coding-system", "MedDRA",
        "--benchmark-dir", str(benchmark_dir),
    ], check=True)

    subprocess.run([
        sys.executable,
        str(ROOT / "scripts" / "build_review_casebook.py"),
        "--predictions", str(benchmark_dir / "predictions.csv"),
        "--explanations", str(benchmark_dir / "explanations.csv"),
        "--records", str(split_path),
        "--output-dir", str(casebook_dir),
        "--max-cases", str(args.casebook_size),
        "--seed", str(args.seed),
    ], check=True)
    write_cadec_report(benchmark_dir)

    summary = {
        "status": "completed",
        "version": "0.0.12",
        "parse_stats": parse_stats,
        "dataset_audit": audit,
        "benchmark_metrics": metrics,
        "paths": {
            "parsed_records": str(parsed_path),
            "split_records": str(split_path),
            "audit_dir": str(audit_dir),
            "benchmark_dir": str(benchmark_dir),
            "benchmark_report": str(benchmark_dir / "report.html"),
            "explanations_html": str(benchmark_dir / "explanations.html"),
            "review_casebook_html": str(casebook_dir / "review_casebook.html"),
            "review_casebook_csv": str(casebook_dir / "review_casebook.csv"),
        },
    }
    (output / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
