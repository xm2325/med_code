#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.analysis import choose_threshold_max_coverage
from cohortcoder.candidate_rationales import build_candidate_rationales
from cohortcoder.core import HistoricalCoder, accuracy_at_k
from cohortcoder.deepseek_real_eval import DeepSeekRealCandidateEvaluator
from cohortcoder.mednorm import (
    assign_cross_dataset_split,
    build_train_derived_terminology,
    fetch_hf_mirror_dataframe,
    mednorm_data_card,
    prepare_mednorm_single_meddra,
)
from cohortcoder.uncertainty import candidate_uncertainty, simple_ood_flag


def load_real_mednorm() -> pd.DataFrame:
    return fetch_hf_mirror_dataframe()


def predict_rows(coder: HistoricalCoder, frame: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    rows, objects = [], []
    for _, record in frame.iterrows():
        prediction = coder.predict_one(str(record["mention"]))
        objects.append(prediction)
        rows.append({
            "record_id": str(record["record_id"]),
            "source_dataset": str(record["source_dataset"]),
            "phrase": str(record["mention"]),
            "gold_code": str(record["gold_code"]),
            "predicted_code": prediction.code,
            "confidence": prediction.confidence,
            "correct": int(prediction.code == str(record["gold_code"])),
            "candidates_json": json.dumps(prediction.candidates),
        })
    return pd.DataFrame(rows), objects


def sample_frame(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(frame) <= n:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def route_case(confidence: float, threshold: float | None, candidates: list[dict]) -> str:
    uncertainty = candidate_uncertainty(candidates)
    if simple_ood_flag(top_score=uncertainty.get("top_score")):
        return "FULL_EXPERT_REVIEW"
    if threshold is not None and float(confidence) >= float(threshold):
        return "AUTO_CANDIDATE"
    return "TOP_K_HUMAN_CHOICE"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Real MedNorm held-out evaluation with optional constrained DeepSeek reranking and grounded rationale for every candidate"
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--train-limit", type=int, default=5000)
    p.add_argument("--validation-limit", type=int, default=300)
    p.add_argument("--baseline-limit", type=int, default=800)
    p.add_argument("--deepseek-limit", type=int, default=30)
    p.add_argument("--deepseek-workers", type=int, default=6)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--target-auto-accuracy", type=float, default=0.95)
    p.add_argument("--model", default="deepseek-v4-pro")
    p.add_argument(
        "--skip-deepseek",
        action="store_true",
        help="Run the exact same real-data baseline/subset and grounded candidate rationales without making an external LLM call.",
    )
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_real_mednorm()
    records = assign_cross_dataset_split(
        prepare_mednorm_single_meddra(raw), test_source="CADEC", val_fraction=0.15
    )
    train_full = records[records.split == "train"].copy()
    val_full = records[records.split == "val"].copy()
    test_full = records[records.split == "test"].copy()

    train = sample_frame(train_full, args.train_limit, args.seed)
    val = sample_frame(val_full, args.validation_limit, args.seed + 1)
    test = sample_frame(test_full, args.baseline_limit, args.seed + 2)
    terminology = build_train_derived_terminology(train_full)
    train_codes = set(train_full.gold_code.astype(str))

    selection = []
    for weight in [0.0, 0.5, 1.0]:
        candidate_coder = HistoricalCoder(history_weight=weight, top_k=args.top_k).fit(train, terminology)
        val_candidate_pred, _ = predict_rows(candidate_coder, val)
        val_candidate_lists = [json.loads(x) for x in val_candidate_pred.candidates_json]
        selection.append({
            "history_weight": weight,
            "val_accuracy_at_1": float(val_candidate_pred.correct.mean()),
            "val_accuracy_at_5": accuracy_at_k(
                val_candidate_pred.gold_code, val_candidate_lists, min(5, args.top_k)
            ),
        })

    selection_df = pd.DataFrame(selection).sort_values(
        ["val_accuracy_at_1", "val_accuracy_at_5", "history_weight"],
        ascending=[False, False, True],
    )
    best_weight = float(selection_df.iloc[0].history_weight)

    coder = HistoricalCoder(history_weight=best_weight, top_k=args.top_k).fit(train, terminology)
    val_pred, _ = predict_rows(coder, val)
    threshold = choose_threshold_max_coverage(val_pred, args.target_auto_accuracy)

    baseline, baseline_objects = predict_rows(coder, test)
    baseline_candidates = [json.loads(x) for x in baseline.candidates_json]
    baseline["gold_seen_in_train"] = baseline.gold_code.astype(str).isin(train_codes)
    baseline["route"] = [
        route_case(row.confidence, threshold, candidates)
        for (_, row), candidates in zip(baseline.iterrows(), baseline_candidates)
    ]

    baseline_metrics = {
        "n_total_real_records": int(len(records)),
        "n_train_full": int(len(train_full)),
        "n_val_full": int(len(val_full)),
        "n_test_full": int(len(test_full)),
        "n_train_model_sample": int(len(train)),
        "n_validation_sample": int(len(val)),
        "n_baseline_test_sample": int(len(test)),
        "sampling_seed": args.seed,
        "test_source": "CADEC",
        "candidate_space": "TRAIN-derived MedDRA code aliases; closed-code diagnostic",
        "selected_history_weight": best_weight,
        "validation_selected_auto_threshold": threshold,
        "baseline_accuracy_at_1": float(baseline.correct.mean()),
        "baseline_accuracy_at_3": accuracy_at_k(
            baseline.gold_code, baseline_candidates, min(3, args.top_k)
        ),
        "baseline_accuracy_at_5": accuracy_at_k(
            baseline.gold_code, baseline_candidates, min(5, args.top_k)
        ),
        "candidate_recall_at_5": accuracy_at_k(
            baseline.gold_code, baseline_candidates, min(5, args.top_k)
        ),
        "gold_seen_in_train_rate": float(baseline.gold_seen_in_train.mean()),
        "seen_code_accuracy_at_1": (
            float(baseline.loc[baseline.gold_seen_in_train, "correct"].mean())
            if baseline.gold_seen_in_train.any() else None
        ),
        "unseen_code_accuracy_at_1": (
            float(baseline.loc[~baseline.gold_seen_in_train, "correct"].mean())
            if (~baseline.gold_seen_in_train).any() else None
        ),
        "route_counts": {str(k): int(v) for k, v in baseline.route.value_counts().to_dict().items()},
        "oracle_top5_human_choice_upper_bound": accuracy_at_k(
            baseline.gold_code, baseline_candidates, min(5, args.top_k)
        ),
        "oracle_warning": (
            "This is only the upper bound if a human always selects the gold code whenever it appears in Top-K; "
            "it is NOT observed human-assisted accuracy."
        ),
    }

    paired_index = baseline.sample(
        n=min(max(args.deepseek_limit, 0), len(baseline)), random_state=args.seed + 3
    ).index.tolist()

    prepared_cases: list[dict[str, Any]] = []
    for idx in paired_index:
        row = baseline.loc[idx]
        prediction = baseline_objects[idx]
        candidates = prediction.candidates[: args.top_k]
        grounding = build_candidate_rationales(
            text=str(row.phrase),
            mention=str(row.phrase),
            candidates=candidates,
            terminology=terminology,
            historical_cases=prediction.historical_cases,
            top_k=args.top_k,
        )
        prepared_cases.append({
            "idx": idx,
            "row": row,
            "prediction": prediction,
            "candidates": candidates,
            "grounding": grounding,
        })

    results_by_idx: dict[int, dict[str, Any] | None] = {int(case["idx"]): None for case in prepared_cases}
    evaluator = None
    if not args.skip_deepseek and prepared_cases:
        evaluator = DeepSeekRealCandidateEvaluator(model=args.model)

        def evaluate_prepared(case: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            row = case["row"]
            result = evaluator.evaluate_case(
                phrase=str(row.phrase),
                candidates=case["candidates"],
                candidate_grounding=case["grounding"],
                allow_external_llm=True,
                data_classification="public",
            )
            return int(case["idx"]), result

        worker_count = max(1, min(int(args.deepseek_workers), len(prepared_cases)))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for idx, result in pool.map(evaluate_prepared, prepared_cases):
                results_by_idx[idx] = result

    case_outputs: list[dict] = []
    ds_rows: list[dict] = []
    for case_input in prepared_cases:
        idx = int(case_input["idx"])
        row = case_input["row"]
        candidates = case_input["candidates"]
        grounding = case_input["grounding"]
        result = results_by_idx[idx]

        ranked_codes = [str(x["code"]) for x in candidates]
        llm_rationales: dict[str, dict] = {}
        api_accepted = bool(result and result.get("accepted"))
        if api_accepted:
            payload = result["payload"]
            ranked_codes = [str(x) for x in payload["ranked_codes"]]
            llm_rationales = {str(x["code"]): x for x in payload["candidate_rationales"]}

        deepseek_top1 = ranked_codes[0] if api_accepted else None
        pipeline_top1 = ranked_codes[0]
        option_rows = []
        for base_rank, (candidate, ground) in enumerate(zip(candidates, grounding), start=1):
            code = str(candidate["code"])
            llm_rat = llm_rationales.get(code, {})
            option_rows.append({
                "code": code,
                "term": str(candidate.get("term", "")),
                "baseline_rank": base_rank,
                "deepseek_rank": ranked_codes.index(code) + 1 if api_accepted else None,
                "model_score": float(candidate.get("score", 0.0) or 0.0),
                "real_evidence_quotes": [str(row.phrase)],
                "deterministic_grounding": ground,
                "deepseek_rationale": str(llm_rat.get("rationale", "")) if api_accepted else None,
                "deepseek_evidence_quotes": llm_rat.get("evidence_quotes", []) if api_accepted else [],
                "display_rationale": (
                    str(llm_rat.get("rationale", ground.get("rationale", "")))
                    if api_accepted else str(ground.get("rationale", ""))
                ),
                "rationale_source": (
                    "deepseek_validated" if api_accepted and code in llm_rationales
                    else "deterministic_grounded"
                ),
            })

        case = {
            "record_id": str(row.record_id),
            "source_dataset": str(row.source_dataset),
            "real_phrase": str(row.phrase),
            "gold_code": str(row.gold_code),
            "gold_seen_in_train": bool(row.gold_seen_in_train),
            "route": str(row.route),
            "baseline_top1": str(row.predicted_code),
            "deepseek_top1": deepseek_top1,
            "pipeline_top1_with_fallback": pipeline_top1,
            "deepseek_call_attempted": bool(evaluator is not None),
            "deepseek_call_accepted": api_accepted,
            "deepseek_validation_errors": result.get("validation_errors", []) if result else [],
            "overall_uncertainty": ((result.get("payload") or {}).get("overall_uncertainty", "") if result else ""),
            "candidate_options": option_rows,
        }
        case_outputs.append(case)
        ds_rows.append({
            "record_id": case["record_id"],
            "gold_code": case["gold_code"],
            "route": case["route"],
            "baseline_top1": case["baseline_top1"],
            "deepseek_top1": case["deepseek_top1"] or "",
            "pipeline_top1_with_fallback": case["pipeline_top1_with_fallback"],
            "baseline_correct": int(case["baseline_top1"] == case["gold_code"]),
            "deepseek_correct": (int(case["deepseek_top1"] == case["gold_code"]) if case["deepseek_top1"] is not None else None),
            "pipeline_correct": int(case["pipeline_top1_with_fallback"] == case["gold_code"]),
            "gold_in_top5": int(case["gold_code"] in [str(x["code"]) for x in candidates]),
            "deepseek_call_attempted": int(case["deepseek_call_attempted"]),
            "deepseek_call_accepted": int(case["deepseek_call_accepted"]),
        })

    ds = pd.DataFrame(ds_rows)
    accepted = ds[ds.deepseek_call_accepted == 1] if len(ds) else pd.DataFrame()
    deepseek_metrics = {
        "execution_mode": "baseline_only" if args.skip_deepseek else "deepseek_requested",
        "n_paired_subset_cases": int(len(ds)),
        "n_deepseek_real_cases": int(ds.deepseek_call_attempted.sum()) if len(ds) else 0,
        "n_valid_deepseek_responses": int(ds.deepseek_call_accepted.sum()) if len(ds) else 0,
        "deepseek_model": args.model if not args.skip_deepseek else None,
        "deepseek_workers": int(args.deepseek_workers) if not args.skip_deepseek else 0,
        "deepseek_api_success_rate": (
            float(ds.deepseek_call_accepted.mean()) if len(ds) and ds.deepseek_call_attempted.any() else None
        ),
        "paired_baseline_accuracy_at_1": float(ds.baseline_correct.mean()) if len(ds) else None,
        "deepseek_reranked_accuracy_at_1_accepted_only": (
            float(accepted.deepseek_correct.mean()) if len(accepted) else None
        ),
        "pipeline_accuracy_at_1_with_deterministic_fallback": (
            float(ds.pipeline_correct.mean()) if len(ds) else None
        ),
        "deepseek_accuracy_delta_accepted_only": (
            float(accepted.deepseek_correct.mean() - accepted.baseline_correct.mean()) if len(accepted) else None
        ),
        "fixed_candidate_recall_at_5": float(ds.gold_in_top5.mean()) if len(ds) else None,
        "oracle_top5_human_choice_upper_bound": float(ds.gold_in_top5.mean()) if len(ds) else None,
        "all_displayed_options_have_real_phrase_evidence": True,
        "rationale_validation_rule": (
            "Every accepted DeepSeek candidate rationale must cite at least one exact allowed real-data phrase and preserve the fixed code set. "
            "When DeepSeek is unavailable, every option still carries deterministic grounded evidence/rationale and is explicitly labelled as such."
        ),
    }

    selection_df.to_csv(out / "model_selection.csv", index=False)
    baseline.to_csv(out / "baseline_real_predictions.csv", index=False)
    ds.to_csv(out / "deepseek_real_predictions.csv", index=False)
    with (out / "deepseek_real_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in case_outputs:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    summary = {
        "release": "0.1.1",
        "dataset": mednorm_data_card(),
        "baseline": baseline_metrics,
        "deepseek": deepseek_metrics,
        "evidence_boundary": (
            "Real phrases and human reference codes come from the public MedNorm-derived data. "
            "Candidate terminology is TRAIN-derived unless an authorised full MedDRA resource is supplied in a separate experiment."
        ),
        "reportability_boundary": (
            "This run is a real-data closed-code evaluation, not a full licensed-MedDRA open-set benchmark."
        ),
    }
    (out / "real_deepseek_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "data_card.json").write_text(
        json.dumps(mednorm_data_card(), indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
