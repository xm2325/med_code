#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def selected_candidate_config(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data["selected_v2"]
    return {
        "history_weight": float(selected["history_weight"]),
        "word_weight": float(selected["word_weight"]),
    }


def predict_frame(coder: AliasAwareHybridCoder, frame: pd.DataFrame) -> pd.DataFrame:
    predictions = coder.predict(frame["mention"].astype(str).tolist(), batch_size=32)
    return pd.DataFrame(
        [
            {
                "record_id": str(record["record_id"]),
                "phrase": str(record["mention"]),
                "gold_code": str(record["gold_code"]),
                "predicted_code": str(pred.code),
                "confidence": float(pred.confidence),
                "correct": int(str(pred.code) == str(record["gold_code"])),
            }
            for (_, record), pred in zip(frame.iterrows(), predictions)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v0.2.2 target-domain policy calibration and fresh CADEC confirmatory evaluation"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-result", default="results/v0.2/candidate_generation_results.json")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--prior-diagnostic-test-limit", type=int, default=500)
    parser.add_argument("--target-policy-calibration-limit", type=int, default=1000)
    parser.add_argument("--confirmatory-test-limit", type=int, default=1000)
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
    cadec_full = records[records.split == "test"].copy().reset_index(drop=True)
    train = sample_frame(train_full, args.train_limit, args.seed)

    # Explicitly exclude the 500-case diagnostic TEST sample already inspected in v0.1.1/v0.2.
    prior_diagnostic = sample_frame(
        cadec_full, args.prior_diagnostic_test_limit, args.seed + 2
    )
    excluded_ids = set(prior_diagnostic["record_id"].astype(str))
    fresh_pool = cadec_full[~cadec_full["record_id"].astype(str).isin(excluded_ids)].copy()

    target_policy_calibration = sample_frame(
        fresh_pool, args.target_policy_calibration_limit, args.seed + 10
    )
    calibration_ids = set(target_policy_calibration["record_id"].astype(str))
    confirmatory_pool = fresh_pool[
        ~fresh_pool["record_id"].astype(str).isin(calibration_ids)
    ].copy()
    confirmatory_test = sample_frame(
        confirmatory_pool, args.confirmatory_test_limit, args.seed + 11
    )

    prior_ids = set(prior_diagnostic["record_id"].astype(str))
    confirmatory_ids = set(confirmatory_test["record_id"].astype(str))
    overlaps = {
        "prior_vs_calibration": len(prior_ids & calibration_ids),
        "prior_vs_confirmatory": len(prior_ids & confirmatory_ids),
        "calibration_vs_confirmatory": len(calibration_ids & confirmatory_ids),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Policy partition overlap detected: {overlaps}")

    terminology = build_train_derived_terminology(train_full, max_aliases_per_code=50)
    coder = AliasAwareHybridCoder(
        history_weight=frozen["history_weight"],
        word_weight=frozen["word_weight"],
        top_k=50,
    ).fit(train, terminology)
    calibration_pred = predict_frame(coder, target_policy_calibration)
    confirmatory_pred = predict_frame(coder, confirmatory_test)

    selected = select_threshold_by_accuracy_lower_bound(
        calibration_pred,
        target_accuracy=args.target_auto_accuracy,
        alpha=args.alpha,
        min_auto=args.min_auto,
    )
    confirmatory = apply_frozen_threshold(confirmatory_pred, selected.threshold)
    confirmatory["target_met"] = bool(
        confirmatory["accuracy_at_1"] is not None
        and confirmatory["accuracy_at_1"] + 1e-12 >= args.target_auto_accuracy
    )

    stress = {}
    for target in [0.90, 0.95, 0.98, 0.99]:
        policy = select_threshold_by_accuracy_lower_bound(
            calibration_pred,
            target_accuracy=target,
            alpha=args.alpha,
            min_auto=args.min_auto,
        )
        test_result = apply_frozen_threshold(confirmatory_pred, policy.threshold)
        test_result["target_met"] = bool(
            test_result["accuracy_at_1"] is not None
            and test_result["accuracy_at_1"] + 1e-12 >= target
        )
        stress[f"{target:.2f}"] = {
            "selection": policy.to_dict(),
            "confirmatory_test": test_result,
        }

    calibration_pred.to_csv(out / "target_policy_calibration_predictions.csv", index=False)
    confirmatory_pred.to_csv(out / "fresh_confirmatory_test_predictions.csv", index=False)
    summary = {
        "release": "0.2.2-target-domain-policy",
        "dataset": mednorm_data_card(),
        "evaluation_design": {
            "n_train_full": int(len(train_full)),
            "n_cadec_full": int(len(cadec_full)),
            "n_train_model_sample": int(len(train)),
            "n_prior_diagnostic_cadec_excluded": int(len(prior_diagnostic)),
            "n_fresh_target_policy_calibration": int(len(target_policy_calibration)),
            "n_fresh_confirmatory_test": int(len(confirmatory_test)),
            "partition_overlaps": overlaps,
            "sampling_seed": int(args.seed),
            "frozen_candidate_config": frozen,
            "prespecified_operational_target": float(args.target_auto_accuracy),
            "one_sided_alpha": float(args.alpha),
            "min_auto": int(args.min_auto),
        },
        "prespecified_95_policy": {
            "target_domain_calibration_selection": selected.to_dict(),
            "fresh_confirmatory_test": confirmatory,
            "release_gate_pass": bool(selected.threshold is not None and confirmatory["target_met"]),
        },
        "policy_stress": stress,
        "release_rule": (
            "The 95% AUTO policy passes only if the threshold is selected without confirmatory labels on fresh target-domain calibration data using the one-sided lower-bound rule, and the locked threshold then reaches at least 95% exact-code agreement on the fresh disjoint confirmatory sample."
        ),
        "interpretation_boundary": [
            "The original 500-case diagnostic TEST sample that motivated this redesign is excluded from both target-domain calibration and confirmatory evaluation.",
            "Candidate-model parameters remain frozen from the earlier non-CADEC validation-only selection; CADEC labels affect routing calibration only, not candidate-model fitting.",
            "A public CADEC confirmation does not substitute for BSRBR-specific temporal and participant-level validation.",
        ],
    }
    (out / "v022_target_domain_policy_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
