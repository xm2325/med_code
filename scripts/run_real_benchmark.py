from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.analysis import write_evaluation_plots
from cohortcoder.realdata import load_records, load_terminology, run_real_benchmark


p = argparse.ArgumentParser()
p.add_argument("--records", required=True)
p.add_argument("--terminology", required=True)
p.add_argument("--output-dir", required=True)
p.add_argument("--target-auto-accuracy", type=float, default=0.95)
p.add_argument("--data-is-synthetic", action="store_true")
p.add_argument("--reference-labels-external", action="store_true")
a = p.parse_args()

metrics = run_real_benchmark(
    load_records(a.records),
    load_terminology(a.terminology),
    a.output_dir,
    target_auto_accuracy=a.target_auto_accuracy,
    external_human_reference=a.reference_labels_external,
    data_is_synthetic=a.data_is_synthetic,
)

output = Path(a.output_dir)
write_evaluation_plots(
    output,
    pd.read_csv(output / "open_set_metrics.csv"),
    pd.read_csv(output / "coverage_accuracy.csv"),
    pd.read_csv(output / "policy_stress_test.csv"),
)
print(metrics)
