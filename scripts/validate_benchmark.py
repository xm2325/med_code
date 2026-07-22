from pathlib import Path
import argparse, json

p = argparse.ArgumentParser()
p.add_argument("--benchmark-dir", required=True)
a = p.parse_args()
root = Path(a.benchmark_dir)
required = ["metrics.json", "results_contract.json", "leakage_audit.json", "experiment_manifest.json", "data_fingerprints.json", "frozen_policy.json"]
missing = [name for name in required if not (root / name).exists()]
if missing:
    raise SystemExit("Missing benchmark artifacts: " + ", ".join(missing))
metrics = json.loads((root / "metrics.json").read_text())
contract = json.loads((root / "results_contract.json").read_text())
print(json.dumps({
    "status": contract.get("status"),
    "reportable": contract.get("reportable"),
    "accuracy_at_1": metrics.get("accuracy_at_1"),
    "auto_candidate_rate": metrics.get("auto_candidate_rate"),
    "auto_candidate_accuracy": metrics.get("auto_candidate_accuracy"),
}, indent=2))
if not contract.get("reportable"):
    print("NOTE: This run is not a formal reportable real-data benchmark.")
