from pathlib import Path
import argparse, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cohortcoder.realdata import parse_cadec

p = argparse.ArgumentParser()
p.add_argument("--cadec-root", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
df = parse_cadec(a.cadec_root, a.output)
print(f"Parsed {len(df):,} rows -> {a.output}")
