from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cohortcoder.realdata import load_records, load_terminology, predict_uncoded

p = argparse.ArgumentParser()
p.add_argument("--historical", required=True)
p.add_argument("--terminology", required=True)
p.add_argument("--input", required=True)
p.add_argument("--frozen-policy", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
policy = json.loads(Path(a.frozen_policy).read_text())
pred = predict_uncoded(load_records(a.historical), load_terminology(a.terminology), load_records(a.input), policy)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
pred.to_csv(a.output, index=False)
print(f"Wrote {len(pred):,} predictions -> {a.output}")
