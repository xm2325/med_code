#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.clinical_context import audit_explanation_context
from cohortcoder.explain import write_explanation_artifacts
from cohortcoder.explanation_quality import apply_explanation_quality_gate, write_explanation_quality_artifacts
from cohortcoder.knowledge import attach_knowledge_provenance, load_terminology_knowledge
from cohortcoder.llm import DeepSeekRationaleClient, apply_deepseek_rationales
from cohortcoder.multilabel import MultiLabelHistoricalCoder
from cohortcoder.multilabel_explain import explain_multilabel_proposals


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one evidence-grounded explanation per proposed MIMIC ICD-10 code")
    parser.add_argument("--records", required=True)
    parser.add_argument("--terminology", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--allow-external-llm", action="store_true")
    parser.add_argument("--data-classification", choices=["public", "synthetic", "restricted", "private"], default="restricted")
    parser.add_argument("--few-shot-csv")
    parser.add_argument("--fallback-top-k", type=int, default=5)
    args = parser.parse_args()

    benchmark = Path(args.benchmark_dir)
    records = pd.read_csv(args.records, dtype=str, keep_default_na=False)
    terminology = load_terminology_knowledge(args.terminology, coding_system="ICD-10")
    predictions = pd.read_csv(benchmark / "predictions.csv", dtype=str, keep_default_na=False)
    policy = json.loads((benchmark / "frozen_policy.json").read_text(encoding="utf-8"))

    train = records[records["split"] == "train"].copy()
    test = records[records["split"] == "test"].copy()
    coder = MultiLabelHistoricalCoder(
        history_weight=float(policy["history_weight"]),
        top_k=100,
    ).fit(train, terminology)
    threshold = policy.get("code_proposal_threshold")
    explanations = explain_multilabel_proposals(
        test,
        predictions,
        terminology,
        coder,
        threshold=float(threshold) if threshold is not None else None,
        fallback_top_k=args.fallback_top_k,
    )
    explanations = audit_explanation_context(explanations)
    explanations = attach_knowledge_provenance(explanations, terminology)

    if args.deepseek:
        few_shot = pd.read_csv(args.few_shot_csv).fillna("").to_dict("records") if args.few_shot_csv else []
        client = DeepSeekRationaleClient(model=args.deepseek_model)
        eligible = [
            item for item in explanations
            if item.get("evidence_quotes") and not item.get("context_review_required", False)
        ]
        enhanced = apply_deepseek_rationales(
            eligible,
            client,
            allow_external_llm=args.allow_external_llm,
            data_classification=args.data_classification,
            few_shot_examples=few_shot,
        ) if eligible else []
        by_key = {(item["record_id"], item["predicted_code"]): item for item in enhanced}
        explanations = [by_key.get((item["record_id"], item["predicted_code"]), item) for item in explanations]

    explanations = apply_explanation_quality_gate(explanations)
    output = benchmark / "explainability"
    output.mkdir(parents=True, exist_ok=True)
    quality = write_explanation_quality_artifacts(output, explanations)
    metrics = write_explanation_artifacts(output, explanations)
    metrics["explanation_unit"] = "record_code_pair"
    metrics["n_code_proposals_explained"] = len(explanations)
    metrics["quality_gate"] = quality
    (output / "mimic_explainability_manifest.json").write_text(
        json.dumps({
            "version": "0.0.13",
            "coding_system": "ICD-10",
            "task_type": "multilabel_document_coding",
            "explanation_unit": "record_code_pair",
            "deepseek_requested": bool(args.deepseek),
            "data_classification": args.data_classification,
            "explanation_quality_gate": "conservative_downgrade_only",
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
