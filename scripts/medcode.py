#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, extra: list[str]) -> int:
    return subprocess.call([sys.executable, str(ROOT / "scripts" / script), *extra])


def main() -> None:
    parser = argparse.ArgumentParser(prog="medcode", description="Unified MedCode research/application CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("cadec", "Run audited CADEC/MedDRA pipeline"),
        ("mimic", "Run audited MIMIC/ICD-10 pipeline"),
        ("review", "Build uncertainty-aware Top-K human review packets"),
        ("feedback-summary", "Summarize expert feedback ledger"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    mapping = {
        "cadec": "run_cadec_v0013.py",
        "mimic": "run_mimic_v0012.py",
        "review": "build_topk_review_packets.py",
        "feedback-summary": "summarize_expert_feedback.py",
    }
    raise SystemExit(_run(mapping[args.command], args.args))


if __name__ == "__main__":
    main()
