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
from cohortcoder.data import load_coding_records
from cohortcoder.explain import explain_predictions, write_explanation_artifacts
from cohortcoder.explanation_quality import apply_explanation_quality_gate, summarize_explanation_quality
from cohortcoder.knowledge import attach_knowledge_provenance, load_terminology_knowledge
from cohortcoder.llm import DeepSeekRationaleClient, apply_deepseek_rationales
from cohortcoder.model_factory import build_singlelabel_coder_from_policy
from cohortcoder.realdata import assign_document_splits
from cohortcoder.source_evidence import apply_task_input_spans


def _attach_test_text(records: pd.DataFrame, predictions: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = records.copy()
    if "split" not in data or not {"train", "val", "test"}.issubset(set(data["split"].astype(str))):
        data = assign_document_splits(data, seed=seed)
    train = data[data["split"] == "train"].copy()
    test = data[data["split"] == "test"].copy()
    test["_occurrence"] = test.groupby("record_id").cumcount()
    pred = predictions.copy()
    pred["_occurrence"] = pred.groupby("record_id").cumcount()
    source_columns = ["record_id", "_occurrence", "text", "mention"]
    for optional in ["spans_json", "start", "end", "annotation_id", "is_discontinuous"]:
        if optional in test.columns:
            source_columns.append(optional)
    merged = pred.merge(
        test[source_columns],
        on=["record_id", "_occurrence"],
        how="left",
        validate="one_to_one",
    )
    if merged["text"].isna().any():
        raise ValueError("Could not align every prediction to its held-out source record")
    return train, merged.drop(columns=["_occurrence"])


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
    parser = argparse.ArgumentParser(description="Create evidence-grounded explanations for a MedCode benchmark")
    parser.add_argument("--records", required=True)
    parser.add_argument("--terminology", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--coding-system")
    parser.add_argument("--device", help="Optional torch/sentence-transformers device for a frozen advanced model")
    parser.add_argument("--deepseek", action="store_true", help="Use DeepSeek only to rewrite the locked-code rationale")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--allow-external-llm", action="store_true")
    parser.add_argument("--data-classification", choices=["public", "synthetic", "restricted", "private"], default="restricted")
    parser.add_argument("--few-shot-csv", help="Optional TRAIN/development rationale examples; never use TEST rationale labels")
    args = parser.parse_args()

    benchmark = Path(args.benchmark_dir)
    predictions = pd.read_csv(benchmark / "predictions.csv").fillna("")
    manifest_path = benchmark / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    seed = int(manifest.get("seed", 42))
    records = load_coding_records(args.records)
    terminology = load_terminology_knowledge(args.terminology, coding_system=args.coding_system)
    train, aligned = _attach_test_text(records, predictions, seed)

    policy = json.loads((benchmark / "frozen_policy.json").read_text(encoding="utf-8"))
    coder = build_singlelabel_coder_from_policy(train, terminology, policy, device=args.device)
    explanations = explain_predictions(aligned, terminology, coder=coder)
    if "spans_json" in aligned.columns:
        explanations = apply_task_input_spans(explanations, aligned)
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

    explanations = apply_explanation_quality_gate(explanations)
    quality = summarize_explanation_quality(explanations)
    (benchmark / "explanation_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    metrics = write_explanation_artifacts(benchmark, explanations)
    metrics["quality_gate"] = quality
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
