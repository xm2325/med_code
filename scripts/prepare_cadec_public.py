from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.cadec import parse_cadec


p = argparse.ArgumentParser(description="Parse CADEC MedDRA annotations with exact BRAT span auditing")
p.add_argument("--cadec-root", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()

df, stats = parse_cadec(a.cadec_root, a.output)
print(json.dumps(stats, indent=2))
print(f"Parsed {len(df):,} rows -> {a.output}")
