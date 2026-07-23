#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.analysis import choose_threshold_max_coverage
from cohortcoder.candidate_generation import AliasAwareHybridCoder
from cohortcoder.mednorm import (
    assign_cross_dataset_split,
    build_train_derived_terminology,
    fetch_hf_mirror_dataframe,
    mednorm_data_card,
    prepare_mednorm_single_meddra,
)
from cohortcoder.selective_policy import (
    apply_frozen_threshold,
    select_threshold_by_accuracy_lower_bound,
)


def sample_frame(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(frame) <= n:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def predict_frame(coder: AliasAwareHybridCoder, frame: pd.DataFrame) -> pd.DataFrame:
    predictions = coder.predict(frame["mention"].astype(str).tolist(), batch_size=32)
    rows = []
    for (_, record), prediction in zip(frame.iterrows(), predictions):
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "phrase": str(record["mention"]),
                "gold_code": str(record["gold_code"]),
                "predicted_code": str(prediction.code),
                "confidence": float(prediction.confidence),
                "correct": int(str(prediction.code) == str(record["gold_code"])),
            }
        )
    return pd.DataFrame(rows)


def selected_candidate_config(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data["selected_v2"]
    return {
        "history_weight": float(selected["history_weight"]),
        "word_weight": float(selected["word_weight"]),
    }


def calibration_snapshot(predictions: pd.DataFrame, threshold: float | None) -> dict:
    result = apply_frozen_threshold(predictions, threshold)
    if threshold is None:
        result["total_n"] = int(len(predictions))
        return result
    accepted = predictions[pd.to_numeric(predictions["confidence"], errors="coerce") >= float(threshold)]
    result["total_n"] = int(len(predictions))
    result["n_errors"] = int((pd.to_numeric(accepted["correct"], errors="coerce").fillna(0) == 0).sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v0.2.1 disjoint policy calibration for conservative AUTO_CANDIDATE routing"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-result", default="results/v0.2/candidate_generation_results.json")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--model-selection-validation-limit", type=int, default=200)
    parser.add_argument("--policy-calibration-limit", type=int, default=1000)
    parser.add_argument("--test-limit", type=int, default=500)
    parser.add_argument("--target-auto-accuracy", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-auto", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen = selected_candidate_config(Path(args.candidate_result))

    raw = fetch_hf_mirror_dataframe()
    records = assign_cross_dataset_split(
        prepare_mednorm_single_meddra(raw), test_source="CADEC", val_fraction=0.15
    )
    train_full = records[records.split == "train"].copy()
    val_full = records[records.split == "val"].copy()
    test_full = records[records.split == "test"].copy()

    train = sample_frame(train_full, args.train_limit, args.seed)
    model_selection_val = sample_frame(
        val_full, args.model_selection_validation_limit, args.seed + 1
    )
    used_ids = set(model_selection_val["record_id"].astype(str))
    policy_pool = val_full[~val_full["record_id"].astype(str).isin(used_ids)].copy()
    policy_calibration = sample_frame(
        policy_pool, args.policy_calibration_limit, args.seed + 4
    )
    test = sample_frame(test_full, args.test_limit, args.seed + 2)

    overlap = set(model_selection_val["record_id"].astype(str)) & set(policy_calibration["record_id"].astype(str))
    if overlap:
        raise RuntimeError("model-selection and policy-calibration records overlap")

    terminology = build_train_derived_terminology(train_full, max_aliases_per_code=50)
    coder = AliasAwareHybridCoder(
        history_weight=frozen["history_weight"],
        word_weight=frozen["word_weight"],
        top_k=50,
    ).fit(train, terminology)

    model_selection_pred = predict_frame(coder, model_selection_val)
    policy_pred = predict_frame(coder, policy_calibration)
    test_pred = predict_frame(coder, test)

    # Reproduce the previous optimistic rule only for diagnosis. It is not the release policy.
    reused_validation_threshold = choose_threshold_max_coverage(
        model_selection_pred, args.target_auto_accuracy
    )
    independent_empirical_threshold = choose_threshold_max_coverage(
        policy_pred, args.target_auto_accuracy
    )
    conservative = select_threshold_by_accuracy_lower_bound(
        policy_pred,
        target_accuracy=args.target_auto_accuracy,
        alpha=args.alpha,
        min_auto=args.min_auto,
    )

    policies = {
        "previous_reused_model_selection_validation": {
            "selection_set": "model_selection_validation",
            "selection": calibration_snapshot(model_selection_pred, reused_validation_threshold),
            "held_out_test": apply_frozen_threshold(test_pred, reused_validation_threshold),
            "release_eligible": False,
            "reason": "Same small validation subset had already been used for candidate-model selection.",
        },
        "independent_empirical_policy_calibration": {
            "selection_set": "disjoint_policy_calibration",
            "selection": calibration_snapshot(policy_pred, independent_empirical_threshold),
            "held_out_test": apply_frozen_threshold(test_pred, independent_empirical_threshold),
            "release_eligible": False,
            "reason": "Independent data, but selection uses empirical accuracy without a sampling-uncertainty margin.",
        },
        "conservative_lower_bound_policy": {
            "selection_set": "disjoint_policy_calibration",
            "selection": conservative.to_dict(),
            "held_out_test": apply_frozen_threshold(test_pred, conservative.threshold),
            "release_eligible": conservative.threshold is not None,
            "reason": "Only candidate AUTO policy: independent policy-calibration data and one-sided binomial lower bound must meet the prespecified target.",
        },
    }

    stress = {}
    for target in [0.90, 0.95, 0.98, 0.99]:
        selected = select_threshold_by_accuracy_lower_bound(
            policy_pred,
            target_accuracy=target,
            alpha=args.alpha,
            min_auto=args.min_auto,
        )
        stress[f"{target:.2f}"] = {
            "selection": selected.to_dict(),
            "held_out_test": apply_frozen_threshold(test_pred, selected.threshold),
        }

    policy_pred.to_csv(out / "policy_calibration_predictions.csv", index=False)
    test_pred.to_csv(out / "policy_test_predictions.csv", index=False)
    summary = {
        "release": "0.2.1-selective-policy",
        "dataset": mednorm_data_card(),
        "evaluation_design": {
            "n_total_real_records": int(len(records)),
            "n_train_full": int(len(train_full)),
            "n_val_full": int(len(val_full)),
            "n_test_full": int(len(test_full)),
            "n_train_model_sample": int(len(train)),
            "n_model_selection_validation": int(len(model_selection_val)),
            "n_disjoint_policy_calibration": int(len(policy_calibration)),
            "n_test_sample": int(len(test)),
            "model_policy_overlap_n": int(len(overlap)),
            "sampling_seed": int(args.seed),
            "test_source": "CADEC",
            "frozen_candidate_config": frozen,
            "target_auto_accuracy": float(args.target_auto_accuracy),
            "one_sided_alpha": float(args.alpha),
            "min_auto": int(args.min_auto),
        },
        "policies": policies,
        "conservative_policy_stress": stress,
        "release_rule": (
            "AUTO_CANDIDATE may be enabled only when a threshold selected on the disjoint policy-calibration set has a one-sided binomial accuracy lower bound at or above the prespecified target. Otherwise AUTO coverage is zero and cases remain human-routed."
        ),
        "interpretation_boundary": [
            "The held-out TEST sample is never used to select or relax an AUTO threshold.",
            "Meeting the calibration lower-bound criterion does not guarantee the same accuracy on every future cohort or distribution.",
            "This is a public-data closed-code diagnostic; BSRBR requires its own temporal and participant-level validation before deployment decisions.",
        ],
    }
    (out / "v021_selective_policy_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
