from __future__ import annotations

from html import escape
from pathlib import Path
import json

import pandas as pd


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_mimic_report(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    metrics = _read_json(output / "metrics.json")
    profile = _read_json(output / "benchmark_profile.json")
    contract = _read_json(output / "results_contract.json")
    leakage = _read_json(output / "leakage_audit.json")
    stress = pd.read_csv(output / "code_proposal_policy_stress.csv") if (output / "code_proposal_policy_stress.csv").exists() else pd.DataFrame()
    novelty = pd.read_csv(output / "open_set_code_recall.csv") if (output / "open_set_code_recall.csv").exists() else pd.DataFrame()
    model_selection = pd.read_csv(output / "model_selection.csv") if (output / "model_selection.csv").exists() else pd.DataFrame()

    warning = (
        "The selective policy is defined over individual ICD code proposals. "
        "It does not estimate the fraction of complete discharge summaries that can be fully auto-coded."
    )
    metric_rows = [
        ("Recall@10", metrics.get("recall_at_10")),
        ("Precision@10", metrics.get("precision_at_10")),
        ("Selective micro precision", metrics.get("selective_micro_precision")),
        ("Selective micro recall", metrics.get("selective_micro_recall")),
        ("Selective micro F1", metrics.get("selective_micro_f1")),
        ("Selective macro F1", metrics.get("selective_macro_f1")),
        ("Mean code proposals per note", metrics.get("selective_mean_code_proposals_per_note")),
        ("Selected history weight", metrics.get("selected_history_weight")),
        ("Historical-memory Recall@10 delta", metrics.get("historical_memory_recall_at_10_delta")),
    ]
    metric_table = pd.DataFrame(metric_rows, columns=["Metric", "Value"])

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>MedCode MIMIC ICD-10 benchmark</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;line-height:1.5}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccc;padding:.5rem;text-align:left}}th{{background:#f4f4f4}}.warning{{padding:1rem;border:1px solid #b66;background:#fff5f2}}code,pre{{white-space:pre-wrap}}</style></head><body>
<h1>MedCode v0.0.11 — MIMIC-IV-Note → ICD-10</h1>
<p><b>Task:</b> {escape(str(profile.get('task_type', 'multilabel_document_coding')))}</p>
<p><b>Results status:</b> {escape(str(contract.get('status', metrics.get('results_status', 'unknown'))))} &nbsp; <b>Reportable:</b> {escape(str(contract.get('reportable', metrics.get('results_reportable'))))}</p>
<div class='warning'><b>Interpretation guard:</b> {escape(warning)}</div>
<h2>Held-out TEST coding metrics</h2>
{metric_table.to_html(index=False)}
<h2>Validation-selected code-proposal policy stress test</h2>
<p>Each threshold is selected on validation only and applied unchanged to TEST.</p>
{stress.to_html(index=False) if not stress.empty else '<p>No stress-test output.</p>'}
<h2>Seen vs unseen ICD-code recall</h2>
{novelty.to_html(index=False) if not novelty.empty else '<p>No open-set output.</p>'}
<h2>Historical-memory model selection</h2>
{model_selection.to_html(index=False) if not model_selection.empty else '<p>No model-selection output.</p>'}
<h2>Leakage audit</h2><pre>{escape(json.dumps(leakage, indent=2))}</pre>
<h2>Next explainability step</h2>
<p>Run <code>scripts/explain_mimic_icd10.py</code> to create one evidence/rationale object per proposed ICD code. Coding performance, faithfulness, and expert plausibility should be evaluated separately.</p>
</body></html>"""
    path = output / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
