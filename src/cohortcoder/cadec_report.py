from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_cadec_report(output_dir: str | Path) -> Path:
    benchmark = Path(output_dir)
    metrics = _read_json(benchmark / "metrics.json")
    contract = _read_json(benchmark / "results_contract.json")
    audit = _read_json(benchmark.parent / "audit" / "dataset_audit.json")
    open_set = _read_csv(benchmark / "open_set_metrics.csv")
    stress = _read_csv(benchmark / "policy_stress_test.csv")
    failures = _read_csv(benchmark / "failure_summary.csv")
    memory = _read_json(benchmark / "historical_memory_value.json")

    metric_rows = [
        ("Accuracy@1", metrics.get("accuracy_at_1")),
        ("Accuracy@5", metrics.get("accuracy_at_5")),
        ("Candidate recall@10", metrics.get("candidate_recall_at_10")),
        ("AUTO candidate rate", metrics.get("auto_candidate_rate")),
        ("AUTO candidate accuracy", metrics.get("auto_candidate_accuracy")),
        ("Human review rate", metrics.get("human_review_rate")),
        ("Selected history weight", metrics.get("selected_history_weight")),
        ("Historical-memory Accuracy@1 delta", metrics.get("historical_memory_accuracy_delta")),
    ]
    metric_table = pd.DataFrame(metric_rows, columns=["Metric", "Value"])

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>MedCode CADEC MedDRA benchmark</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;line-height:1.5}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccc;padding:.5rem;text-align:left}}th{{background:#f4f4f4}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f7f7;padding:.8rem}}.warning{{padding:1rem;border:1px solid #b66;background:#fff5f2}}</style></head><body>
<h1>MedCode v0.0.12 — CADEC → MedDRA</h1>
<p><b>Results status:</b> {escape(str(contract.get('status', metrics.get('results_status', 'unknown'))))} &nbsp; <b>Reportable:</b> {escape(str(contract.get('reportable', metrics.get('results_reportable'))))}</p>
<div class='warning'><b>Interpretation guard:</b> AUTO/HUMAN_REVIEW thresholds are selected on validation only. TEST coverage/accuracy curves are descriptive and must not be used to retune the policy.</div>
<h2>Pre-flight dataset readiness</h2><pre>{escape(json.dumps(audit, indent=2)) if audit else 'No v0.0.12 audit artifact found.'}</pre>
<h2>Held-out TEST coding metrics</h2>{metric_table.to_html(index=False)}
<h2>Seen vs unseen codes</h2>{open_set.to_html(index=False) if not open_set.empty else '<p>No open-set output.</p>'}
<h2>Validation-selected workload policy stress test</h2>{stress.to_html(index=False) if not stress.empty else '<p>No stress-test output.</p>'}
<h2>Failure taxonomy</h2>{failures.to_html(index=False) if not failures.empty else '<p>No failure output.</p>'}
<h2>Historical coding memory</h2><pre>{escape(json.dumps(memory, indent=2))}</pre>
<h2>Explainability and expert review</h2>
<p>After the end-to-end runner completes, inspect <code>explanations.html</code> for code/evidence/why cards and <code>../casebook/review_casebook.html</code> for priority expert-review cases with original context.</p>
</body></html>"""
    path = benchmark / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
