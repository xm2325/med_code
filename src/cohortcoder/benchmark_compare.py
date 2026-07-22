from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _first_json(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        if path.exists():
            return _read_json(path)
    return {}


def _explainability_metrics(root: Path) -> dict[str, Any]:
    return _first_json([
        root / "explainability_metrics.json",
        root / "explainability" / "explainability_metrics.json",
    ])


def _plausibility_metrics(root: Path) -> dict[str, Any]:
    return _first_json([
        root / "rationale_review_summary.json",
        root / "explainability" / "rationale_review_summary.json",
        root / "expert_review_summary.json",
    ])


def _rationale_overlap_metrics(root: Path) -> dict[str, Any]:
    return _first_json([
        root / "rationale_overlap_metrics.json",
        root / "rationale_eval" / "rationale_overlap_metrics.json",
        root / "explainability" / "rationale_eval" / "rationale_overlap_metrics.json",
    ])


def _positive_rate(summary: dict[str, Any], field: str) -> float | None:
    value = summary.get(field)
    return value.get("positive_rate") if isinstance(value, dict) else None


def build_dual_benchmark_summary(cadec_dir: str | Path, mimic_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Create a four-axis side-by-side summary without pooling incompatible tasks.

    The shared reporting axes are coding, faithfulness, plausibility, and workload/policy.
    Their values remain separate; no composite score is calculated.
    """
    cadec = Path(cadec_dir)
    mimic = Path(mimic_dir)
    cadec_metrics = _read_json(cadec / "metrics.json")
    mimic_metrics = _read_json(mimic / "metrics.json")
    cadec_contract = _read_json(cadec / "results_contract.json")
    mimic_contract = _read_json(mimic / "results_contract.json")
    cadec_exp = _explainability_metrics(cadec)
    mimic_exp = _explainability_metrics(mimic)
    cadec_plaus = _plausibility_metrics(cadec)
    mimic_plaus = _plausibility_metrics(mimic)
    cadec_overlap = _rationale_overlap_metrics(cadec)
    mimic_overlap = _rationale_overlap_metrics(mimic)

    rows = [
        {
            "benchmark": "CADEC -> MedDRA",
            "task_type": "single_label_concept_normalization",
            "coding_primary": "Accuracy@1",
            "coding_primary_value": cadec_metrics.get("accuracy_at_1"),
            "coding_secondary": "Accuracy@5",
            "coding_secondary_value": cadec_metrics.get("accuracy_at_5"),
            "workload_policy": "AUTO candidate accuracy / coverage",
            "workload_precision_or_accuracy": cadec_metrics.get("auto_candidate_accuracy"),
            "workload_coverage_or_recall": cadec_metrics.get("auto_candidate_rate"),
            "grounded_rate": cadec_exp.get("grounded_rate"),
            "verbatim_evidence_rate": cadec_exp.get("verbatim_evidence_rate"),
            "mean_sufficiency_gap": cadec_exp.get("mean_sufficiency_gap"),
            "mean_comprehensiveness_drop": cadec_exp.get("mean_comprehensiveness_drop"),
            "expert_rationale_correct_rate": _positive_rate(cadec_plaus, "expert_rationale_correct"),
            "expert_evidence_complete_rate": _positive_rate(cadec_plaus, "expert_evidence_complete"),
            "gold_rationale_macro_char_f1": cadec_overlap.get("macro_char_f1"),
            "reportable": cadec_contract.get("reportable"),
        },
        {
            "benchmark": "MIMIC-IV-Note -> ICD-10",
            "task_type": "multilabel_document_coding",
            "coding_primary": "Micro F1 at validation-selected code-proposal threshold",
            "coding_primary_value": mimic_metrics.get("selective_micro_f1"),
            "coding_secondary": "Recall@10",
            "coding_secondary_value": mimic_metrics.get("recall_at_10"),
            "workload_policy": "Individual code-proposal precision / recall",
            "workload_precision_or_accuracy": mimic_metrics.get("selective_micro_precision"),
            "workload_coverage_or_recall": mimic_metrics.get("selective_micro_recall"),
            "grounded_rate": mimic_exp.get("grounded_rate"),
            "verbatim_evidence_rate": mimic_exp.get("verbatim_evidence_rate"),
            "mean_sufficiency_gap": mimic_exp.get("mean_sufficiency_gap"),
            "mean_comprehensiveness_drop": mimic_exp.get("mean_comprehensiveness_drop"),
            "expert_rationale_correct_rate": _positive_rate(mimic_plaus, "expert_rationale_correct"),
            "expert_evidence_complete_rate": _positive_rate(mimic_plaus, "expert_evidence_complete"),
            "gold_rationale_macro_char_f1": mimic_overlap.get("macro_char_f1"),
            "reportable": mimic_contract.get("reportable"),
        },
    ]
    table = pd.DataFrame(rows)
    summary = {
        "version": "0.0.11",
        "pooled_score": None,
        "pooling_prohibited_reason": (
            "CADEC is single-label mention/concept normalization, while MIMIC is multi-label "
            "document coding. Coding, faithfulness, plausibility, and workload metrics must "
            "also remain separate rather than being collapsed into one composite score."
        ),
        "evaluation_axes": {
            "coding": "task-specific held-out prediction performance",
            "faithfulness": "grounding plus retain/remove model-score diagnostics",
            "plausibility": "expert review and/or human rationale-span overlap where available",
            "workload": "validation-selected operational policy evaluated unchanged on TEST",
        },
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
<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;line-height:1.5}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #ccc;padding:.45rem;text-align:left;vertical-align:top}th{background:#f4f4f4}.warning{padding:1rem;border:1px solid #c88;background:#fff7f3}</style></head><body>
<h1>MedCode v0.0.11 dual benchmark</h1>
<div class='warning'><b>No pooled score.</b> CADEC and MIMIC are different coding tasks, and coding accuracy, faithfulness, plausibility, and workload are distinct outcomes.</div>
""" + table.to_html(index=False) + """
<h2>Four evaluation axes</h2>
<ol><li><b>Coding:</b> task-specific held-out prediction metrics.</li><li><b>Faithfulness:</b> verbatim grounding and evidence retain/remove effects on the model score.</li><li><b>Plausibility:</b> expert review and/or human rationale-span overlap where available.</li><li><b>Workload:</b> validation-selected policy evaluated unchanged on TEST.</li></ol>
</body></html>"""
    (output / "dual_benchmark_summary.html").write_text(html, encoding="utf-8")
    return summary
