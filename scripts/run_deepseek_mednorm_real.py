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

from cohortcoder.analysis import choose_threshold_max_coverage
from cohortcoder.candidate_rationales import build_candidate_rationales
from cohortcoder.core import HistoricalCoder, accuracy_at_k
from cohortcoder.deepseek_real_eval import DeepSeekRealCandidateEvaluator
from cohortcoder.mednorm import assign_cross_dataset_split, build_train_derived_terminology, fetch_hf_mirror_rows, mednorm_data_card, prepare_mednorm_single_meddra
from cohortcoder.uncertainty import candidate_uncertainty, simple_ood_flag

RAW_TSV = "https://huggingface.co/datasets/awacke1/MedNorm2SnomedCT2UMLS/resolve/main/mednorm_full.tsv"


def load_real_mednorm() -> pd.DataFrame:
    try:
        return pd.read_csv(RAW_TSV, sep="\t", dtype=str, keep_default_na=False)
    except Exception:
        return fetch_hf_mirror_rows()


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
    p = argparse.ArgumentParser(description="Real MedNorm held-out evaluation with constrained DeepSeek reranking and rationale for every candidate")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--train-limit", type=int, default=5000)
    p.add_argument("--validation-limit", type=int, default=300)
    p.add_argument("--baseline-limit", type=int, default=800)
    p.add_argument("--deepseek-limit", type=int, default=30)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--target-auto-accuracy", type=float, default=0.95)
    p.add_argument("--model", default="deepseek-v4-pro")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_real_mednorm()
    records = assign_cross_dataset_split(prepare_mednorm_single_meddra(raw), test_source="CADEC", val_fraction=0.15)
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
        coder = HistoricalCoder(history_weight=weight, top_k=args.top_k).fit(train, terminology)
        val_pred, _ = predict_rows(coder, val)
        val_candidates = [json.loads(x) for x in val_pred.candidates_json]
        selection.append({
            "history_weight": weight,
            "val_accuracy_at_1": float(val_pred.correct.mean()),
            "val_accuracy_at_5": accuracy_at_k(val_pred.gold_code, val_candidates, min(5, args.top_k)),
        })
    selection_df = pd.DataFrame(selection).sort_values(["val_accuracy_at_1", "val_accuracy_at_5", "history_weight"], ascending=[False, False, True])
    best_weight = float(selection_df.iloc[0].history_weight)
    coder = HistoricalCoder(history_weight=best_weight, top_k=args.top_k).fit(train, terminology)
    val_pred, _ = predict_rows(coder, val)
    threshold = choose_threshold_max_coverage(val_pred, args.target_auto_accuracy)
    baseline, baseline_objects = predict_rows(coder, test)
    baseline_candidates = [json.loads(x) for x in baseline.candidates_json]
    baseline["gold_seen_in_train"] = baseline.gold_code.astype(str).isin(train_codes)
    baseline["route"] = [route_case(row.confidence, threshold, candidates) for (_, row), candidates in zip(baseline.iterrows(), baseline_candidates)]

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
        "baseline_accuracy_at_3": accuracy_at_k(baseline.gold_code, baseline_candidates, min(3, args.top_k)),
        "baseline_accuracy_at_5": accuracy_at_k(baseline.gold_code, baseline_candidates, min(5, args.top_k)),
        "candidate_recall_at_5": accuracy_at_k(baseline.gold_code, baseline_candidates, min(5, args.top_k)),
        "gold_seen_in_train_rate": float(baseline.gold_seen_in_train.mean()),
        "seen_code_accuracy_at_1": float(baseline.loc[baseline.gold_seen_in_train, "correct"].mean()) if baseline.gold_seen_in_train.any() else None,
        "unseen_code_accuracy_at_1": float(baseline.loc[~baseline.gold_seen_in_train, "correct"].mean()) if (~baseline.gold_seen_in_train).any() else None,
        "route_counts": baseline.route.value_counts().to_dict(),
        "oracle_top5_human_choice_upper_bound": accuracy_at_k(baseline.gold_code, baseline_candidates, min(5, args.top_k)),
        "oracle_warning": "This is only the upper bound if a human always selects the gold code whenever it appears in Top-K; it is NOT observed human-assisted accuracy.",
    }

    deepseek_index = baseline.sample(n=min(args.deepseek_limit, len(baseline)), random_state=args.seed + 3).index.tolist()
    evaluator = DeepSeekRealCandidateEvaluator(model=args.model)
    case_outputs, ds_rows = [], []
    for idx in deepseek_index:
        row = baseline.loc[idx]
        prediction = baseline_objects[idx]
        candidates = prediction.candidates[: args.top_k]
        grounding = build_candidate_rationales(
            text=str(row.phrase), mention=str(row.phrase), candidates=candidates,
            terminology=terminology, historical_cases=prediction.historical_cases, top_k=args.top_k,
        )
        result = evaluator.evaluate_case(
            phrase=str(row.phrase), candidates=candidates, candidate_grounding=grounding,
            allow_external_llm=True, data_classification="public",
        )
        if result["accepted"]:
            payload = result["payload"]
            ranked_codes = [str(x) for x in payload["ranked_codes"]]
            llm_rationales = {str(x["code"]): x for x in payload["candidate_rationales"]}
            deepseek_top1 = ranked_codes[0]
        else:
            ranked_codes = [str(x["code"]) for x in candidates]
            llm_rationales = {}
            deepseek_top1 = ranked_codes[0]
        option_rows = []
        for base_rank, (candidate, ground) in enumerate(zip(candidates, grounding), start=1):
            code = str(candidate["code"])
            llm_rat = llm_rationales.get(code, {})
            option_rows.append({
                "code": code,
                "term": str(candidate.get("term", "")),
                "baseline_rank": base_rank,
                "deepseek_rank": ranked_codes.index(code) + 1,
                "model_score": float(candidate.get("score", 0.0) or 0.0),
                "real_evidence_quotes": [str(row.phrase)],
                "deterministic_grounding": ground,
                "deepseek_rationale": str(llm_rat.get("rationale", ground.get("rationale", ""))),
                "deepseek_evidence_quotes": llm_rat.get("evidence_quotes", [str(row.phrase)]),
                "rationale_source": "deepseek_validated" if code in llm_rationales else "deterministic_fallback",
            })
        case = {
            "record_id": str(row.record_id), "source_dataset": str(row.source_dataset), "real_phrase": str(row.phrase),
            "gold_code": str(row.gold_code), "gold_seen_in_train": bool(row.gold_seen_in_train),
            "route": str(row.route), "baseline_top1": str(row.predicted_code), "deepseek_top1": deepseek_top1,
            "deepseek_call_accepted": bool(result["accepted"]), "deepseek_validation_errors": result["validation_errors"],
            "overall_uncertainty": (result.get("payload") or {}).get("overall_uncertainty", ""), "candidate_options": option_rows,
        }
        case_outputs.append(case)
        ds_rows.append({
            "record_id": case["record_id"], "gold_code": case["gold_code"], "route": case["route"],
            "baseline_top1": case["baseline_top1"], "deepseek_top1": case["deepseek_top1"],
            "baseline_correct": int(case["baseline_top1"] == case["gold_code"]),
            "deepseek_correct": int(case["deepseek_top1"] == case["gold_code"]),
            "gold_in_top5": int(case["gold_code"] in ranked_codes),
            "deepseek_call_accepted": int(case["deepseek_call_accepted"]),
        })

    ds = pd.DataFrame(ds_rows)
    deepseek_metrics = {
        "n_deepseek_real_cases": int(len(ds)),
        "deepseek_model": args.model,
        "deepseek_api_success_rate": float(ds.deepseek_call_accepted.mean()) if len(ds) else None,
        "paired_baseline_accuracy_at_1": float(ds.baseline_correct.mean()) if len(ds) else None,
        "deepseek_reranked_accuracy_at_1": float(ds.deepseek_correct.mean()) if len(ds) else None,
        "deepseek_accuracy_delta": float(ds.deepseek_correct.mean() - ds.baseline_correct.mean()) if len(ds) else None,
        "fixed_candidate_recall_at_5": float(ds.gold_in_top5.mean()) if len(ds) else None,
        "oracle_top5_human_choice_upper_bound": float(ds.gold_in_top5.mean()) if len(ds) else None,
        "all_displayed_options_have_real_phrase_evidence": True,
        "rationale_validation_rule": "Every accepted DeepSeek candidate rationale must cite at least one exact allowed real-data phrase and preserve the fixed code set.",
    }

    selection_df.to_csv(out / "model_selection.csv", index=False)
    baseline.to_csv(out / "baseline_real_predictions.csv", index=False)
    ds.to_csv(out / "deepseek_real_predictions.csv", index=False)
    with (out / "deepseek_real_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in case_outputs:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    summary = {
        "release": "0.1.1", "dataset": mednorm_data_card(), "baseline": baseline_metrics, "deepseek": deepseek_metrics,
        "evidence_boundary": "Real phrases and human reference codes come from the public MedNorm-derived data. Candidate terminology is TRAIN-derived unless an authorised full MedDRA resource is supplied in a separate experiment.",
        "reportability_boundary": "This run is a real-data closed-code evaluation, not a full licensed-MedDRA open-set benchmark.",
    }
    (out / "real_deepseek_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "data_card.json").write_text(json.dumps(mednorm_data_card(), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
