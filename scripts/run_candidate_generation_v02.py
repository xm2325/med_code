#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.candidate_generation import AliasAwareHybridCoder
from cohortcoder.core import HistoricalCoder, accuracy_at_k
from cohortcoder.mednorm import (
    assign_cross_dataset_split,
    build_train_derived_terminology,
    fetch_hf_mirror_dataframe,
    mednorm_data_card,
    prepare_mednorm_single_meddra,
)


def sample_frame(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(frame) <= n:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def predict_rows(coder, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[list[dict]]]:
    rows: list[dict] = []
    candidate_lists: list[list[dict]] = []
    for _, record in frame.iterrows():
        prediction = coder.predict_one(str(record["mention"]))
        candidates = prediction.candidates
        candidate_lists.append(candidates)
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "source_dataset": str(record["source_dataset"]),
                "phrase": str(record["mention"]),
                "gold_code": str(record["gold_code"]),
                "predicted_code": str(prediction.code),
                "confidence": float(prediction.confidence),
                "correct": int(str(prediction.code) == str(record["gold_code"])),
                "candidates_json": json.dumps(candidates, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows), candidate_lists


def reciprocal_rank(gold: pd.Series, candidate_lists: list[list[dict]], max_k: int = 50) -> float:
    values: list[float] = []
    for gold_code, candidates in zip(gold.astype(str), candidate_lists):
        rank = next(
            (idx for idx, item in enumerate(candidates[:max_k], start=1) if str(item["code"]) == gold_code),
            None,
        )
        values.append(0.0 if rank is None else 1.0 / rank)
    return float(np.mean(values)) if values else float("nan")


def metric_block(
    pred: pd.DataFrame,
    candidates: list[list[dict]],
    *,
    train_codes: set[str],
    train_aliases: dict[str, set[str]],
) -> dict:
    gold = pred["gold_code"].astype(str)
    seen = gold.isin(train_codes)
    exact_alias_seen = pd.Series(
        [
            str(phrase).casefold().strip() in train_aliases.get(str(code), set())
            for phrase, code in zip(pred["phrase"], gold)
        ],
        index=pred.index,
    )

    def subgroup(mask: pd.Series) -> dict:
        idx = np.flatnonzero(mask.to_numpy())
        if not len(idx):
            return {"n": 0, "accuracy_at_1": None, "recall_at_5": None, "recall_at_20": None, "recall_at_50": None}
        sub = pred.iloc[idx]
        sub_candidates = [candidates[i] for i in idx]
        return {
            "n": int(len(sub)),
            "accuracy_at_1": float(sub["correct"].mean()),
            "recall_at_5": accuracy_at_k(sub["gold_code"], sub_candidates, 5),
            "recall_at_20": accuracy_at_k(sub["gold_code"], sub_candidates, 20),
            "recall_at_50": accuracy_at_k(sub["gold_code"], sub_candidates, 50),
        }

    return {
        "n": int(len(pred)),
        "accuracy_at_1": float(pred["correct"].mean()),
        "recall_at_5": accuracy_at_k(gold, candidates, 5),
        "recall_at_20": accuracy_at_k(gold, candidates, 20),
        "recall_at_50": accuracy_at_k(gold, candidates, 50),
        "mrr_at_50": reciprocal_rank(gold, candidates, 50),
        "candidate_generation_failures_at_5": int(
            sum(str(g) not in [str(x["code"]) for x in cs[:5]] for g, cs in zip(gold, candidates))
        ),
        "candidate_generation_failures_at_20": int(
            sum(str(g) not in [str(x["code"]) for x in cs[:20]] for g, cs in zip(gold, candidates))
        ),
        "candidate_generation_failures_at_50": int(
            sum(str(g) not in [str(x["code"]) for x in cs[:50]] for g, cs in zip(gold, candidates))
        ),
        "seen_code": subgroup(seen),
        "unseen_code": subgroup(~seen),
        "exact_gold_alias_seen_in_train": subgroup(exact_alias_seen),
        "novel_gold_wording": subgroup(~exact_alias_seen),
    }


def train_alias_map(train_full: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for code, group in train_full.groupby("gold_code"):
        result[str(code)] = {
            str(value).casefold().strip()
            for value in group["mention"].astype(str)
            if str(value).strip()
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v0.2 real MedNorm candidate-generation benchmark with alias-aware retrieval"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--validation-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = fetch_hf_mirror_dataframe()
    records = assign_cross_dataset_split(
        prepare_mednorm_single_meddra(raw), test_source="CADEC", val_fraction=0.15
    )
    train_full = records[records.split == "train"].copy()
    val_full = records[records.split == "val"].copy()
    test_full = records[records.split == "test"].copy()

    train = sample_frame(train_full, args.train_limit, args.seed)
    val = sample_frame(val_full, args.validation_limit, args.seed + 1)
    test = sample_frame(test_full, args.test_limit, args.seed + 2)
    terminology = build_train_derived_terminology(train_full, max_aliases_per_code=50)
    train_codes = set(train_full["gold_code"].astype(str))
    alias_map = train_alias_map(train_full)

    # Reproduce the v0.1.1 family without using TEST for model selection.
    baseline_selection: list[dict] = []
    for history_weight in [0.0, 0.5, 1.0]:
        coder = HistoricalCoder(history_weight=history_weight, top_k=args.top_k).fit(train, terminology)
        pred, candidates = predict_rows(coder, val)
        baseline_selection.append(
            {
                "model": "v0.1.1_single_term",
                "history_weight": history_weight,
                "word_weight": None,
                "val_accuracy_at_1": float(pred.correct.mean()),
                "val_recall_at_5": accuracy_at_k(pred.gold_code, candidates, 5),
                "val_recall_at_20": accuracy_at_k(pred.gold_code, candidates, 20),
            }
        )
    baseline_sel = pd.DataFrame(baseline_selection).sort_values(
        ["val_accuracy_at_1", "val_recall_at_5", "history_weight"],
        ascending=[False, False, True],
    )
    baseline_weight = float(baseline_sel.iloc[0]["history_weight"])

    # v0.2 selection explicitly optimises candidate recall on VALIDATION, then top-1 accuracy.
    v2_selection: list[dict] = []
    for history_weight in [0.25, 0.5, 0.75]:
        for word_weight in [0.15, 0.35]:
            coder = AliasAwareHybridCoder(
                history_weight=history_weight,
                word_weight=word_weight,
                top_k=args.top_k,
            ).fit(train, terminology)
            pred, candidates = predict_rows(coder, val)
            v2_selection.append(
                {
                    "model": "v0.2_alias_aware_hybrid",
                    "history_weight": history_weight,
                    "word_weight": word_weight,
                    "val_accuracy_at_1": float(pred.correct.mean()),
                    "val_recall_at_5": accuracy_at_k(pred.gold_code, candidates, 5),
                    "val_recall_at_20": accuracy_at_k(pred.gold_code, candidates, 20),
                }
            )
    v2_sel = pd.DataFrame(v2_selection).sort_values(
        ["val_recall_at_5", "val_recall_at_20", "val_accuracy_at_1", "history_weight", "word_weight"],
        ascending=[False, False, False, True, True],
    )
    best = v2_sel.iloc[0]

    baseline_coder = HistoricalCoder(history_weight=baseline_weight, top_k=args.top_k).fit(train, terminology)
    baseline_pred, baseline_candidates = predict_rows(baseline_coder, test)
    baseline_metrics = metric_block(
        baseline_pred,
        baseline_candidates,
        train_codes=train_codes,
        train_aliases=alias_map,
    )

    v2_coder = AliasAwareHybridCoder(
        history_weight=float(best["history_weight"]),
        word_weight=float(best["word_weight"]),
        top_k=args.top_k,
    ).fit(train, terminology)
    v2_pred, v2_candidates = predict_rows(v2_coder, test)
    v2_metrics = metric_block(
        v2_pred,
        v2_candidates,
        train_codes=train_codes,
        train_aliases=alias_map,
    )

    comparison = {
        "accuracy_at_1_delta": v2_metrics["accuracy_at_1"] - baseline_metrics["accuracy_at_1"],
        "recall_at_5_delta": v2_metrics["recall_at_5"] - baseline_metrics["recall_at_5"],
        "recall_at_20_delta": v2_metrics["recall_at_20"] - baseline_metrics["recall_at_20"],
        "recall_at_50_delta": v2_metrics["recall_at_50"] - baseline_metrics["recall_at_50"],
        "mrr_at_50_delta": v2_metrics["mrr_at_50"] - baseline_metrics["mrr_at_50"],
    }

    merged = baseline_pred[["record_id", "phrase", "gold_code", "predicted_code", "correct", "candidates_json"]].rename(
        columns={
            "predicted_code": "baseline_predicted_code",
            "correct": "baseline_correct",
            "candidates_json": "baseline_candidates_json",
        }
    ).merge(
        v2_pred[["record_id", "predicted_code", "correct", "candidates_json"]].rename(
            columns={
                "predicted_code": "v2_predicted_code",
                "correct": "v2_correct",
                "candidates_json": "v2_candidates_json",
            }
        ),
        on="record_id",
        how="inner",
    )
    merged.to_csv(out / "candidate_generation_predictions.csv", index=False)
    pd.concat([baseline_sel, v2_sel], ignore_index=True).to_csv(out / "model_selection.csv", index=False)

    failure_rows: list[dict] = []
    for i, row in merged.iterrows():
        b_codes = [str(x["code"]) for x in json.loads(row["baseline_candidates_json"])[:5]]
        v_codes = [str(x["code"]) for x in json.loads(row["v2_candidates_json"])[:5]]
        gold = str(row["gold_code"])
        if gold not in b_codes or gold not in v_codes:
            failure_rows.append(
                {
                    "record_id": str(row["record_id"]),
                    "phrase": str(row["phrase"]),
                    "gold_code": gold,
                    "baseline_gold_in_top5": gold in b_codes,
                    "v2_gold_in_top5": gold in v_codes,
                    "baseline_top5": b_codes,
                    "v2_top5": v_codes,
                }
            )
    with (out / "candidate_failure_examples.jsonl").open("w", encoding="utf-8") as handle:
        for item in failure_rows[:100]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "release": "0.2.0-candidate-generation",
        "dataset": mednorm_data_card(),
        "evaluation_design": {
            "n_total_real_records": int(len(records)),
            "n_train_full": int(len(train_full)),
            "n_val_full": int(len(val_full)),
            "n_test_full": int(len(test_full)),
            "n_train_model_sample": int(len(train)),
            "n_validation_sample": int(len(val)),
            "n_test_sample": int(len(test)),
            "sampling_seed": int(args.seed),
            "test_source": "CADEC",
            "candidate_space": "TRAIN-derived MedDRA code aliases; closed-code diagnostic",
            "selection_rule": "VALIDATION Recall@5, then Recall@20, then Accuracy@1; TEST never used for model selection",
        },
        "selected_baseline": {"history_weight": baseline_weight},
        "selected_v2": {
            "history_weight": float(best["history_weight"]),
            "word_weight": float(best["word_weight"]),
        },
        "baseline_v0_1_1_family": baseline_metrics,
        "v0_2_alias_aware_hybrid": v2_metrics,
        "delta_v0_2_minus_baseline": comparison,
        "interpretation_boundary": [
            "This benchmark isolates candidate generation on public real data; it does not claim clinical deployment readiness.",
            "The candidate dictionary is TRAIN-derived, so unseen TEST codes remain structurally unavailable.",
            "Recall@K measures whether the reference code is present in the candidate menu; it is not human-assisted accuracy.",
            "All configuration selection is performed on VALIDATION only before the held-out CADEC-derived TEST sample is scored.",
        ],
    }
    (out / "candidate_generation_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
