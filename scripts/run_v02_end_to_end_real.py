#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.analysis import choose_threshold_max_coverage
from cohortcoder.candidate_generation import AliasAwareHybridCoder
from cohortcoder.candidate_rationales import build_candidate_rationales
from cohortcoder.core import accuracy_at_k
from cohortcoder.deepseek_real_eval import DeepSeekRealCandidateEvaluator
from cohortcoder.mednorm import (
    assign_cross_dataset_split,
    build_train_derived_terminology,
    fetch_hf_mirror_dataframe,
    mednorm_data_card,
    prepare_mednorm_single_meddra,
)
from cohortcoder.uncertainty import candidate_uncertainty, simple_ood_flag


def sample_frame(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(frame) <= n:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def load_frozen_candidate_config(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected_v2", {})
    history_weight = float(selected["history_weight"])
    word_weight = float(selected["word_weight"])
    return {"history_weight": history_weight, "word_weight": word_weight}


def predict_frame(coder: AliasAwareHybridCoder, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[Any]]:
    predictions = coder.predict(frame["mention"].astype(str).tolist(), batch_size=32)
    rows: list[dict[str, Any]] = []
    for (_, record), prediction in zip(frame.iterrows(), predictions):
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "source_dataset": str(record["source_dataset"]),
                "phrase": str(record["mention"]),
                "gold_code": str(record["gold_code"]),
                "predicted_code": str(prediction.code),
                "confidence": float(prediction.confidence),
                "correct": int(str(prediction.code) == str(record["gold_code"])),
                "candidates_json": json.dumps(prediction.candidates, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows), predictions


def route_case(confidence: float, threshold: float | None, candidates: list[dict[str, Any]]) -> str:
    uncertainty = candidate_uncertainty(candidates)
    if simple_ood_flag(top_score=uncertainty.get("top_score")):
        return "FULL_EXPERT_REVIEW"
    if threshold is not None and float(confidence) >= float(threshold):
        return "AUTO_CANDIDATE"
    return "TOP_K_HUMAN_CHOICE"


def route_metrics(pred: pd.DataFrame, candidate_lists: list[list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for route, group in pred.groupby("route", sort=True):
        idx = group.index.tolist()
        candidates = [candidate_lists[i] for i in idx]
        out[str(route)] = {
            "n": int(len(group)),
            "coverage": float(len(group) / len(pred)) if len(pred) else None,
            "accuracy_at_1": float(group["correct"].mean()) if len(group) else None,
            "recall_at_5": accuracy_at_k(group["gold_code"], candidates, 5) if len(group) else None,
        }
    return out


def evaluate_deepseek_case(
    *,
    idx: int,
    row: pd.Series,
    prediction: Any,
    terminology: pd.DataFrame,
    evaluator: DeepSeekRealCandidateEvaluator,
    top_k: int,
) -> tuple[int, dict[str, Any]]:
    candidates = prediction.candidates[:top_k]
    grounding = build_candidate_rationales(
        text=str(row.phrase),
        mention=str(row.phrase),
        candidates=candidates,
        terminology=terminology,
        historical_cases=prediction.historical_cases,
        top_k=top_k,
    )
    result = evaluator.evaluate_case(
        phrase=str(row.phrase),
        candidates=candidates,
        candidate_grounding=grounding,
        allow_external_llm=True,
        data_classification="public",
        max_attempts=3,
    )
    accepted = bool(result["accepted"])
    baseline_codes = [str(item["code"]) for item in candidates]
    ranked_codes = baseline_codes
    llm_rationales: dict[str, dict[str, Any]] = {}
    if accepted:
        payload = result["payload"]
        ranked_codes = [str(code) for code in payload["ranked_codes"]]
        llm_rationales = {
            str(item["code"]): dict(item) for item in payload["candidate_rationales"]
        }

    options: list[dict[str, Any]] = []
    for baseline_rank, (candidate, ground) in enumerate(zip(candidates, grounding), start=1):
        code = str(candidate["code"])
        llm = llm_rationales.get(code, {})
        options.append(
            {
                "code": code,
                "term": str(candidate.get("term", "")),
                "baseline_rank": baseline_rank,
                "deepseek_rank": ranked_codes.index(code) + 1 if accepted else None,
                "matched_alias": str(candidate.get("matched_alias", "")),
                "model_score": float(candidate.get("score", 0.0) or 0.0),
                "real_evidence_quotes": [str(row.phrase)],
                "deterministic_grounding": ground,
                "display_rationale": (
                    str(llm.get("rationale", ground.get("rationale", "")))
                    if accepted
                    else str(ground.get("rationale", ""))
                ),
                "rationale_source": "deepseek_validated" if accepted and code in llm_rationales else "deterministic_grounded",
                "deepseek_evidence_quotes": llm.get("evidence_quotes", []) if accepted else [],
            }
        )

    deepseek_top1 = ranked_codes[0] if accepted else None
    fallback_top1 = deepseek_top1 or baseline_codes[0]
    return idx, {
        "record_id": str(row.record_id),
        "source_dataset": str(row.source_dataset),
        "real_phrase": str(row.phrase),
        "gold_code": str(row.gold_code),
        "route": str(row.route),
        "retrieval_top1": str(row.predicted_code),
        "deepseek_top1": deepseek_top1,
        "pipeline_top1_with_fallback": fallback_top1,
        "deepseek_call_accepted": accepted,
        "deepseek_validation_errors": result.get("validation_errors", []),
        "overall_uncertainty": ((result.get("payload") or {}).get("overall_uncertainty", "")),
        "gold_in_top5": str(row.gold_code) in baseline_codes,
        "candidate_options": options,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.2 real MedNorm/CADEC end-to-end candidate generation + DeepSeek evaluation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-result", default="results/v0.2/candidate_generation_results.json")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--validation-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=500)
    parser.add_argument("--deepseek-limit", type=int, default=50)
    parser.add_argument("--deepseek-workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-top-k", type=int, default=50)
    parser.add_argument("--target-auto-accuracy", type=float, default=0.95)
    parser.add_argument("--model", default="deepseek-v4-pro")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_candidate_config(Path(args.candidate_result))

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

    coder = AliasAwareHybridCoder(
        history_weight=frozen["history_weight"],
        word_weight=frozen["word_weight"],
        top_k=args.retrieval_top_k,
    ).fit(train, terminology)

    val_pred, _ = predict_frame(coder, val)
    threshold = choose_threshold_max_coverage(val_pred, args.target_auto_accuracy)
    test_pred, test_objects = predict_frame(coder, test)
    candidate_lists = [obj.candidates for obj in test_objects]
    test_pred["route"] = [
        route_case(row.confidence, threshold, candidates)
        for (_, row), candidates in zip(test_pred.iterrows(), candidate_lists)
    ]

    retrieval_metrics = {
        "n": int(len(test_pred)),
        "accuracy_at_1": float(test_pred.correct.mean()),
        "recall_at_5": accuracy_at_k(test_pred.gold_code, candidate_lists, 5),
        "recall_at_20": accuracy_at_k(test_pred.gold_code, candidate_lists, 20),
        "recall_at_50": accuracy_at_k(test_pred.gold_code, candidate_lists, 50),
        "validation_selected_auto_threshold": threshold,
        "target_auto_accuracy": float(args.target_auto_accuracy),
        "routes": route_metrics(test_pred, candidate_lists),
    }

    paired_indices = test_pred.sample(
        n=min(max(args.deepseek_limit, 0), len(test_pred)), random_state=args.seed + 3
    ).index.tolist()
    evaluator = DeepSeekRealCandidateEvaluator(model=args.model)
    case_outputs: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.deepseek_workers)) as pool:
        futures = {
            pool.submit(
                evaluate_deepseek_case,
                idx=idx,
                row=test_pred.loc[idx],
                prediction=test_objects[idx],
                terminology=terminology,
                evaluator=evaluator,
                top_k=args.top_k,
            ): idx
            for idx in paired_indices
        }
        for future in as_completed(futures):
            idx, case = future.result()
            case_outputs[idx] = case

    cases = [case_outputs[idx] for idx in paired_indices]
    paired = pd.DataFrame(
        [
            {
                "record_id": case["record_id"],
                "gold_code": case["gold_code"],
                "route": case["route"],
                "retrieval_top1": case["retrieval_top1"],
                "deepseek_top1": case["deepseek_top1"] or "",
                "pipeline_top1_with_fallback": case["pipeline_top1_with_fallback"],
                "retrieval_correct": int(case["retrieval_top1"] == case["gold_code"]),
                "deepseek_correct": int(case["deepseek_top1"] == case["gold_code"]) if case["deepseek_top1"] else None,
                "pipeline_correct": int(case["pipeline_top1_with_fallback"] == case["gold_code"]),
                "gold_in_top5": int(case["gold_in_top5"]),
                "deepseek_call_accepted": int(case["deepseek_call_accepted"]),
            }
            for case in cases
        ]
    )
    accepted = paired[paired.deepseek_call_accepted == 1]
    corrected = int(((paired.retrieval_correct == 0) & (paired.pipeline_correct == 1)).sum())
    harmed = int(((paired.retrieval_correct == 1) & (paired.pipeline_correct == 0)).sum())
    deepseek_metrics = {
        "n_paired_subset_cases": int(len(paired)),
        "deepseek_model": args.model,
        "deepseek_workers": int(args.deepseek_workers),
        "n_valid_deepseek_responses": int(paired.deepseek_call_accepted.sum()),
        "deepseek_valid_response_rate": float(paired.deepseek_call_accepted.mean()) if len(paired) else None,
        "retrieval_accuracy_at_1": float(paired.retrieval_correct.mean()) if len(paired) else None,
        "deepseek_accuracy_at_1_accepted_only": float(accepted.deepseek_correct.mean()) if len(accepted) else None,
        "pipeline_accuracy_at_1_with_fallback": float(paired.pipeline_correct.mean()) if len(paired) else None,
        "pipeline_delta_vs_retrieval": float(paired.pipeline_correct.mean() - paired.retrieval_correct.mean()) if len(paired) else None,
        "fixed_candidate_recall_at_5": float(paired.gold_in_top5.mean()) if len(paired) else None,
        "cases_corrected_by_deepseek": corrected,
        "cases_harmed_by_deepseek": harmed,
        "net_corrected_minus_harmed": corrected - harmed,
    }

    test_pred.to_csv(out / "v02_real_predictions.csv", index=False)
    paired.to_csv(out / "v02_deepseek_paired_predictions.csv", index=False)
    with (out / "v02_deepseek_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    summary = {
        "release": "0.2.0-end-to-end",
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
            "candidate_config_source": str(args.candidate_result),
            "candidate_config_frozen_from_validation_only": frozen,
            "deepseek_subset_size": int(len(paired)),
        },
        "v0_2_retrieval": retrieval_metrics,
        "v0_2_deepseek": deepseek_metrics,
        "rationale_contract": {
            "every_displayed_option_has_real_phrase_evidence": True,
            "accepted_llm_output_preserves_fixed_candidate_set": True,
            "accepted_llm_output_requires_one_rationale_per_candidate": True,
            "invalid_llm_output_uses_deterministic_grounded_fallback": True,
        },
        "interpretation_boundary": [
            "This remains a TRAIN-derived closed-code public-data benchmark, not a licensed full-MedDRA open-set evaluation.",
            "The candidate configuration was frozen from validation-only selection before this held-out TEST evaluation.",
            "The DeepSeek comparison is paired on a fixed-seed subset; larger repeated evaluations are required before claiming stable superiority.",
            "Recall@K is candidate availability, not observed human-assisted accuracy.",
        ],
    }
    (out / "v02_end_to_end_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
