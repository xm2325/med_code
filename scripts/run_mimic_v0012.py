#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.artifact_metadata import stamp_benchmark_artifacts
from cohortcoder.knowledge import load_terminology_knowledge
from cohortcoder.mimic_audit import write_mimic_audit_artifacts
from cohortcoder.mimic_benchmark import run_mimic_icd10_benchmark
from cohortcoder.mimic_report import write_mimic_report


def main() -> None:
    p = argparse.ArgumentParser(description="Run audited MIMIC-IV-Note -> ICD-10 benchmark end to end")
    p.add_argument("--records", required=True)
    p.add_argument("--terminology", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-proposal-precision", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--source-version", default="MIMIC-IV-Note 2.2 + MIMIC-IV 2.2")
    p.add_argument("--audit-sample-size", type=int, default=30)
    p.add_argument("--allow-audit-failures", action="store_true")
    args = p.parse_args()

    output = Path(args.output_dir)
    audit_dir = output / "audit"
    benchmark_dir = output / "benchmark"
    for path in [audit_dir, benchmark_dir]:
        path.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(args.records, dtype=str, keep_default_na=False)
    terminology = load_terminology_knowledge(args.terminology, coding_system="ICD-10")
    audit = write_mimic_audit_artifacts(
        records,
        terminology,
        audit_dir,
        sample_size=args.audit_sample_size,
    )
    if not audit["ready_for_benchmark"] and not args.allow_audit_failures:
        (output / "pipeline_summary.json").write_text(json.dumps({
            "status": "blocked_by_dataset_audit",
            "dataset_audit": audit,
        }, indent=2), encoding="utf-8")
        raise SystemExit("MIMIC audit failed. Review audit/dataset_audit.json before benchmarking.")

    metrics = run_mimic_icd10_benchmark(
        records,
        terminology,
        benchmark_dir,
        target_proposal_precision=args.target_proposal_precision,
        external_human_reference=True,
        data_is_synthetic=False,
        source_version=args.source_version,
        batch_size=args.batch_size,
    )
    stamp_benchmark_artifacts(
        benchmark_dir,
        version="0.0.12",
        benchmark_profile="mimic_iv_note_icd10_multilabel",
    )
    write_mimic_report(benchmark_dir)

    explain_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "explain_mimic_icd10.py"),
        "--records", str(args.records),
        "--terminology", str(args.terminology),
        "--benchmark-dir", str(benchmark_dir),
    ]
    subprocess.run(explain_cmd, check=True)

    summary = {
        "status": "completed",
        "version": "0.0.12",
        "dataset_audit": audit,
        "benchmark_metrics": metrics,
        "paths": {
            "audit_dir": str(audit_dir),
            "benchmark_dir": str(benchmark_dir),
            "benchmark_report": str(benchmark_dir / "report.html"),
            "explanations_html": str(benchmark_dir / "explainability" / "explanations.html"),
        },
    }
    (output / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
