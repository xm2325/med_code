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
from cohortcoder.core import HistoricalCoder
from cohortcoder.data import load_coding_records
from cohortcoder.explain import explain_predictions, write_explanation_artifacts
from cohortcoder.knowledge import attach_knowledge_provenance, load_terminology_knowledge
from cohortcoder.llm import DeepSeekRationaleClient, apply_deepseek_rationales
from cohortcoder.realdata import predict_uncoded


def _apply_llm_only_when_grounded(explanations, client, *, allow_external_llm, data_classification, few_shot_examples):
    eligible = [item for item in explanations if item.get("evidence_quotes") and not item.get("context_review_required", False)]
    enhanced = apply_deepseek_rationales(
        eligible,
        client,
        allow_external_llm=allow_external_llm,
        data_classification=data_classification,
        few_shot_examples=few_shot_examples,
    ) if eligible else []
    by_key = {(item["record_id"], item["predicted_code"]): item for item in enhanced}
    return [by_key.get((item["record_id"], item["predicted_code"]), item) for item in explanations]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict codes for new records and produce grounded explanations")
    parser.add_argument("--historical", required=True)
    parser.add_argument("--terminology", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--frozen-policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coding-system")
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--allow-external-llm", action="store_true")
    parser.add_argument("--data-classification", choices=["public", "synthetic", "restricted", "private"], default="restricted")
    parser.add_argument("--few-shot-csv")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    historical = load_coding_records(args.historical)
    new_records = load_coding_records(args.input)
    terminology = load_terminology_knowledge(args.terminology, coding_system=args.coding_system)
    policy = json.loads(Path(args.frozen_policy).read_text(encoding="utf-8"))

    predictions = predict_uncoded(historical, terminology, new_records, policy)
    predictions = predictions.rename(columns={"decision": "policy_decision"})
    aligned = pd.concat(
        [
            predictions.rename(columns={"policy_decision": "decision"}).reset_index(drop=True),
            new_records[["text", "mention"]].reset_index(drop=True),
        ],
        axis=1,
    )
    coder = HistoricalCoder(history_weight=float(policy["history_weight"]), top_k=10).fit(historical, terminology)
    explanations = explain_predictions(aligned, terminology, coder=coder)
    explanations = attach_knowledge_provenance(explanations, terminology)
    explanations = audit_explanation_context(explanations)

    if args.deepseek:
        few_shot = pd.read_csv(args.few_shot_csv).fillna("").to_dict("records") if args.few_shot_csv else []
        client = DeepSeekRationaleClient(model=args.deepseek_model)
        explanations = _apply_llm_only_when_grounded(
            explanations,
            client,
            allow_external_llm=args.allow_external_llm,
            data_classification=args.data_classification,
            few_shot_examples=few_shot,
        )

    # The frozen policy decision is retained for audit, but a failed explanation-context
    # guard may only make the operational decision more conservative, never less.
    predictions["final_decision"] = [item["decision"] for item in explanations]
    predictions["explanation_status"] = [item["explanation_status"] for item in explanations]
    predictions["context_review_required"] = [bool(item.get("context_review_required", False)) for item in explanations]
    predictions.to_csv(output / "predictions.csv", index=False)

    metrics = write_explanation_artifacts(output, explanations)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
