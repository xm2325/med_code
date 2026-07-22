from pathlib import Path
import argparse, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cohortcoder.realdata import load_records, assign_document_splits, assert_document_disjoint

p = argparse.ArgumentParser()
p.add_argument("--records", required=True)
p.add_argument("--output", required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--train-fraction", type=float, default=0.70)
p.add_argument("--val-fraction", type=float, default=0.15)
a = p.parse_args()
df = assign_document_splits(load_records(a.records), seed=a.seed, train=a.train_fraction, val=a.val_fraction)
assert_document_disjoint(df)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.output, index=False)
print(df["split"].value_counts().to_string())
