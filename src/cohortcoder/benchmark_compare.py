from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _explainability_metrics(root: Path) -> dict[str, Any]:
    candidates = [
        root / "explainability_metrics.json",
        root / "explainability" / "explainability_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            return _read_json(path)
    return {}


def build_dual_benchmark_summary(cadec_dir: str | Path, mimic_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Create a side-by-side summary while preserving different task definitions.

    CADEC concept normalization and MIMIC multi-label ICD coding do not share a single
    valid primary metric, so this function intentionally does not compute a pooled score.
    """
    cadec = Path(cadec_dir)
    mimic = Path(mimic_dir)
    cadec_metrics = _read_json(cadec / "metrics.json")
    mimic_metrics = _read_json(mimic / "metrics.json")
    cadec_contract = _read_json(cadec / "results_contract.json")
    mimic_contract = _read_json(mimic / "results_contract.json")
    cadec_exp = _explainability_metrics(cadec)
    mimic_exp = _explainability_metrics(mimic)

    rows = [
        {
            "benchmark": "CADEC -> MedDRA",
            "task_type": "single_label_concept_normalization",
            "coding_metric_1": "Accuracy@1",
            "coding_metric_1_value": cadec_metrics.get("accuracy_at_1"),
            "coding_metric_2": "Accuracy@5",
            "coding_metric_2_value": cadec_metrics.get("accuracy_at_5"),
            "selective_metric": "AUTO candidate accuracy",
            "selective_metric_value": cadec_metrics.get("auto_candidate_accuracy"),
            "grounded_rate": cadec_exp.get("grounded_rate"),
            "verbatim_evidence_rate": cadec_exp.get("verbatim_evidence_rate"),
            "reportable": cadec_contract.get("reportable"),
        },
        {
            "benchmark": "MIMIC-IV-Note -> ICD-10",
            "task_type": "multilabel_document_coding",
            "coding_metric_1": "Micro F1 (validation-selected code proposal threshold)",
            "coding_metric_1_value": mimic_metrics.get("selective_micro_f1"),
            "coding_metric_2": "Recall@10",
            "coding_metric_2_value": mimic_metrics.get("recall_at_10"),
            "selective_metric": "Per-code proposal precision",
            "selective_metric_value": mimic_metrics.get("selective_micro_precision"),
            "grounded_rate": mimic_exp.get("grounded_rate"),
            "verbatim_evidence_rate": mimic_exp.get("verbatim_evidence_rate"),
            "reportable": mimic_contract.get("reportable"),
        },
    ]
    table = pd.DataFrame(rows)
    summary = {
        "version": "0.0.11",
        "pooled_score": None,
        "pooling_prohibited_reason": (
            "CADEC is single-label mention/concept normalization, while MIMIC is multi-label "
            "document coding. Task-specific coding metrics must be interpreted separately."
        ),
        "shared_explainability_questions": [
            "Are evidence spans verbatim and grounded in the source text?",
            "Does removing evidence reduce the selected-code score?",
            "Do human reviewers judge the evidence and rationale clinically appropriate?",
        ],
        "benchmarks": rows,
    }
    return summary, table


def write_dual_benchmark_summary(cadec_dir: str | Path, mimic_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary, table = build_dual_benchmark_summary(cadec_dir, mimic_dir)
    (output / "dual_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table.to_csv(output / "dual_benchmark_summary.csv", index=False)
    html = """<!doctype html><html><head><meta charset='utf-8'><title>MedCode dual benchmark</title>
<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;line-height:1.5}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.5rem;text-align:left}th{background:#f4f4f4}.warning{padding:1rem;border:1px solid #c88;background:#fff7f3}</style></head><body>
<h1>MedCode v0.0.11 dual benchmark</h1>
<div class='warning'><b>Do not pool these coding metrics.</b> CADEC is single-label concept normalization; MIMIC is multi-label document coding.</div>
""" + table.to_html(index=False) + """
<h2>Shared explainability evaluation</h2>
<ul><li>verbatim/grounded evidence</li><li>faithfulness via evidence retention/removal</li><li>expert plausibility review or gold rationale overlap where available</li></ul>
</body></html>"""
    (output / "dual_benchmark_summary.html").write_text(html, encoding="utf-8")
    return summary
