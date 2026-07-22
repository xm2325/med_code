#!/usr/bin/env python
from __future__ import annotations

import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.audit_replay import build_audit_bundle, validate_audit_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-dir", required=True)
    p.add_argument("--release", default="0.0.19")
    a = p.parse_args()
    d = Path(a.benchmark_dir)
    files = {
        "metrics": d / "metrics.json",
        "predictions": d / "predictions.csv",
        "frozen_policy": d / "frozen_policy.json",
        "results_contract": d / "results_contract.json",
        "experiment_manifest": d / "experiment_manifest.json",
    }
    if (d / "explanation_quality.json").exists(): files["explanation_quality"] = d / "explanation_quality.json"
    bundle_path = d / "audit_bundle.json"
    build_audit_bundle(bundle_path, release=a.release, files=files, decision_semantics={
        "AUTO_CANDIDATE":"model candidate accepted only under frozen policy and safety gates",
        "TOP_K_HUMAN_CHOICE":"human selects among grounded candidate options or escalates",
        "FULL_EXPERT_REVIEW":"no automatic coding claim",
    })
    print(json.dumps(validate_audit_bundle(bundle_path), indent=2))

if __name__ == "__main__": main()
