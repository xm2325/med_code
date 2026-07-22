#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.core import HistoricalCoder
from cohortcoder.data import load_coding_records
from cohortcoder.explain import explain_predictions, write_explanation_artifacts
from cohortcoder.knowledge import load_terminology_knowledge
from cohortcoder.llm import DeepSeekRationaleClient, apply_deepseek_rationales
from cohortcoder.realdata import assign_document_splits


def _attach_test_text(records: pd.DataFrame, predictions: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = records.copy()
    if "split" not in data or not {"train", "val", "test"}.issubset(set(data["split"].astype(str))):
        data = assign_document_splits(data, seed=seed)
    train = data[data["split"] == "train"].copy()
    test = data[data["split"] == "test"].copy()
    test["_occurrence"] = test.groupby("record_id").cumcount()
    pred = predictions.copy()
    pred["_occurrence"] = pred.groupby("record_id").cumcount()
    merged = pred.merge(
        test[["record_id", "_occurrence", "text", "mention"]],
        on=["record_id", "_occurrence"],
        how="left",
        validate="one_to_one",
    )
    if merged["text"].isna().any():
        raise ValueError("Could not align every prediction to its held-out source record")
    return train, merged.drop(columns=["_occurrence"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create evidence-grounded explanations for a MedCode benchmark")
    parser.add_argument("--records", required=True)
    parser.add_argument("--terminology", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--coding-system")
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
    coder = HistoricalCoder(history_weight=float(policy["history_weight"]), top_k=10).fit(train, terminology)
    explanations = explain_predictions(aligned, terminology, coder=coder)

    if args.deepseek:
        few_shot = pd.read_csv(args.few_shot_csv).fillna("").to_dict("records") if args.few_shot_csv else []
        client = DeepSeekRationaleClient(model=args.deepseek_model)
        explanations = apply_deepseek_rationales(
            explanations,
            client,
            allow_external_llm=args.allow_external_llm,
            data_classification=args.data_classification,
            few_shot_examples=few_shot,
        )

    metrics = write_explanation_artifacts(benchmark, explanations)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
