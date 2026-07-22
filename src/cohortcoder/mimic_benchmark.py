from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
from typing import Any, Iterable

import pandas as pd

from .benchmark_profiles import MIMIC_IV_ICD10
from .mimic import assert_subject_disjoint
from .multilabel import (
    MultiLabelHistoricalCoder,
    parse_code_list,
    ranking_metrics,
    select_threshold_max_recall_at_precision,
    threshold_metrics,
)
from .multilabel_batch import rank_dataframe_batched
from .results import contract_from_benchmark_metadata, write_results_contract


def _dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _all_train_codes(train: pd.DataFrame) -> set[str]:
    return {code for value in train["gold_codes_json"] for code in parse_code_list(value)}


def novelty_recall(predictions: pd.DataFrame, seen_codes: set[str], *, k: int = 10) -> pd.DataFrame:
    rows = []
    totals = {"seen_code": [0, 0], "unseen_code": [0, 0]}
    for _, row in predictions.iterrows():
        gold = set(parse_code_list(row["gold_codes_json"]))
        candidates = json.loads(str(row["candidates_json"]))[: int(k)]
        predicted = {str(item.get("code", "")) for item in candidates}
        for code in gold:
            group = "seen_code" if code in seen_codes else "unseen_code"
            totals[group][1] += 1
            totals[group][0] += int(code in predicted)
    for group, (hits, total) in totals.items():
        rows.append({
            "subgroup": group,
            "gold_code_count": int(total),
            f"recall_at_{k}": float(hits / total) if total else None,
        })
    return pd.DataFrame(rows)


def _policy_stress(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    targets: Iterable[float] = (0.90, 0.95, 0.98, 0.99),
) -> pd.DataFrame:
    rows = []
    for target in targets:
        threshold = select_threshold_max_recall_at_precision(
            validation_predictions,
            target_precision=float(target),
        )
        if threshold is None:
            rows.append({
                "target_precision": float(target),
                "validation_threshold": None,
                "test_micro_precision": None,
                "test_micro_recall": 0.0,
                "test_micro_f1": 0.0,
                "test_mean_code_proposals_per_note": 0.0,
                "target_met_on_test": False,
            })
            continue
        metrics = threshold_metrics(test_predictions, threshold)
        rows.append({
            "target_precision": float(target),
            "validation_threshold": float(threshold),
            "test_micro_precision": metrics["micro_precision"],
            "test_micro_recall": metrics["micro_recall"],
            "test_micro_f1": metrics["micro_f1"],
            "test_mean_code_proposals_per_note": metrics["mean_code_proposals_per_note"],
            "target_met_on_test": bool(float(metrics["micro_precision"]) + 1e-12 >= float(target)),
        })
    return pd.DataFrame(rows)


def run_mimic_icd10_benchmark(
    records: pd.DataFrame,
    terminology: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_proposal_precision: float = 0.95,
    external_human_reference: bool = True,
    data_is_synthetic: bool = False,
    source_version: str = "MIMIC-IV-Note 2.2 + MIMIC-IV 2.2",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run the patient-disjoint multi-label ICD-10 benchmark.

    The threshold policy applies to individual code proposals. It must not be described
    as the fraction of entire discharge summaries that can be automatically coded.
    """
    required = {"record_id", "subject_id", "text", "gold_codes_json", "split"}
    if not required.issubset(records.columns):
        raise ValueError(f"MIMIC benchmark records require {sorted(required)}")
    if not {"code", "term"}.issubset(terminology.columns):
        raise ValueError("terminology requires code and term")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    assert_subject_disjoint(records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train = records[records["split"] == "train"].copy()
    val = records[records["split"] == "val"].copy()
    test = records[records["split"] == "test"].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("train/val/test must all be non-empty")

    # All candidate history weights share identical terminology/history vector spaces
    # and code centroids. Build these indexes once, then change only the fusion weight.
    coder = MultiLabelHistoricalCoder(history_weight=0.0, top_k=100).fit(train, terminology)
    tuning_rows = []
    for history_weight in [0.0, 0.25, 0.50, 0.75, 1.0]:
        coder.history_weight = float(history_weight)
        val_predictions_for_weight = rank_dataframe_batched(coder, val, batch_size=batch_size)
        weight_metrics = ranking_metrics(val_predictions_for_weight, k_values=(5, 10, 20))
        tuning_rows.append({"history_weight": history_weight, **weight_metrics})
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(output / "model_selection.csv", index=False)
    best = tuning.sort_values(
        ["recall_at_10", "precision_at_10", "history_weight"],
        ascending=[False, False, True],
    ).iloc[0]

    coder.history_weight = float(best.history_weight)
    val_predictions = rank_dataframe_batched(coder, val, batch_size=batch_size)
    test_predictions = rank_dataframe_batched(coder, test, batch_size=batch_size)
    coder.history_weight = 0.0
    baseline_test = rank_dataframe_batched(coder, test, batch_size=batch_size)
    coder.history_weight = float(best.history_weight)

    threshold = select_threshold_max_recall_at_precision(
        val_predictions,
        target_precision=target_proposal_precision,
    )
    ranking = ranking_metrics(test_predictions, k_values=(5, 10, 20))
    if threshold is None:
        selective = {
            "threshold": None,
            "micro_precision": None,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "exact_match": 0.0,
            "n_code_proposals": 0,
            "mean_code_proposals_per_note": 0.0,
        }
    else:
        selective = threshold_metrics(test_predictions, float(threshold))

    baseline_ranking = ranking_metrics(baseline_test, k_values=(5, 10, 20))
    seen_codes = _all_train_codes(train)
    novelty = novelty_recall(test_predictions, seen_codes, k=10)
    stress = _policy_stress(val_predictions, test_predictions)

    train_subjects = set(train["subject_id"].astype(str))
    val_subjects = set(val["subject_id"].astype(str))
    test_subjects = set(test["subject_id"].astype(str))
    subject_overlap = {
        "train_val": len(train_subjects & val_subjects),
        "train_test": len(train_subjects & test_subjects),
        "val_test": len(val_subjects & test_subjects),
    }
    contract = contract_from_benchmark_metadata(
        external_human_reference=external_human_reference,
        group_disjoint_test=not any(subject_overlap.values()),
        candidate_dictionary_source="authorised_icd_dictionary",
        test_used_for_selection_or_tuning=False,
        data_is_synthetic=data_is_synthetic,
        provenance_recorded=True,
    )

    metrics: dict[str, Any] = {
        **ranking,
        **{f"selective_{key}": value for key, value in selective.items()},
        "selected_history_weight": float(best.history_weight),
        "selected_code_proposal_threshold": threshold,
        "target_code_proposal_precision": float(target_proposal_precision),
        "terminology_only_recall_at_10": baseline_ranking.get("recall_at_10"),
        "historical_memory_recall_at_10_delta": float(ranking.get("recall_at_10", 0.0) - baseline_ranking.get("recall_at_10", 0.0)),
        "results_status": contract.status,
        "results_reportable": contract.reportable,
        "task_type": MIMIC_IV_ICD10.task_type,
        "policy_unit": "individual_code_proposal",
        "full_note_automation_claim_allowed": False,
        "batch_size": int(batch_size),
    }

    test_predictions.to_csv(output / "predictions.csv", index=False)
    val_predictions.to_csv(output / "validation_predictions.csv", index=False)
    baseline_test.to_csv(output / "terminology_only_test_predictions.csv", index=False)
    novelty.to_csv(output / "open_set_code_recall.csv", index=False)
    stress.to_csv(output / "code_proposal_policy_stress.csv", index=False)
    _dump(output / "metrics.json", metrics)
    _dump(output / "benchmark_profile.json", MIMIC_IV_ICD10.to_dict())
    _dump(output / "frozen_policy.json", {
        "version": "0.0.11",
        "task_type": MIMIC_IV_ICD10.task_type,
        "history_weight": float(best.history_weight),
        "code_proposal_threshold": threshold,
        "target_code_proposal_precision": float(target_proposal_precision),
        "policy_unit": "individual_code_proposal",
        "full_note_automation_claim_allowed": False,
    })
    _dump(output / "leakage_audit.json", {
        "split_unit": "subject_id",
        "subject_overlap": subject_overlap,
    })
    _dump(output / "experiment_manifest.json", {
        "version": "0.0.11",
        "benchmark_profile": MIMIC_IV_ICD10.name,
        "source_version": source_version,
        "external_human_reference": bool(external_human_reference),
        "data_is_synthetic": bool(data_is_synthetic),
        "test_used_for_selection_or_tuning": False,
        "batch_size": int(batch_size),
        "index_reuse_during_weight_selection": True,
        "data_governance": "credentialed_source_and_sensitive_derivatives",
    })
    _dump(output / "data_fingerprints.json", {
        "records_sha256": sha256(records.to_csv(index=False).encode()).hexdigest(),
        "terminology_sha256": sha256(terminology.to_csv(index=False).encode()).hexdigest(),
    })
    write_results_contract(output / "results_contract.json", contract)
    return metrics
