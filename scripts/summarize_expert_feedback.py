#!/usr/bin/env python
from __future__ import annotations

import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.feedback import feedback_summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feedback-jsonl", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = [json.loads(line) for line in Path(a.feedback_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = feedback_summary(rows)
    Path(a.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
