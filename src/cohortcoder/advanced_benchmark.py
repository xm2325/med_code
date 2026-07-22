from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
from typing import Any

import pandas as pd

from .advanced import (
    AdvancedModelConfig,
    AdvancedSingleLabelCoder,
    CrossEncoderCandidateReranker,
    DenseSemanticIndex,
)
from .analysis import (
    annotate_prediction_diagnostics,
    choose_threshold_max_coverage,
    coverage_accuracy_curve,
    failure_summary,
    policy_stress_test,
    subgroup_metrics,
)
from .core import HistoricalCoder, accuracy_at_k
from .realdata import assign_document_splits, assert_document_disjoint
from .results import contract_from_benchmark_metadata, write_results_contract


def _dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _predict_frame(coder: Any, df: pd.DataFrame) -> pd.DataFrame:
    query = df["mention"].where(df["mention"].astype(str).str.len() > 0, df["text"])
    predictions = coder.predict(query)
    rows = []
    for (_, record), prediction in zip(df.iterrows(), predictions):
        rows.append({
            "record_id": str(record.record_id),
            "gold_code": str(record.gold_code),
            "gold_term": str(record.gold_term),
            "predicted_code": prediction.code,
            "predicted_term": prediction.term,
            "confidence": prediction.confidence,
            "correct": int(prediction.code == str(record.gold_code)),
            "candidates_json": json.dumps(prediction.candidates),
            "historical_cases_json": json.dumps(prediction.historical_cases),
        })
    return pd.DataFrame(rows)


def _score_validation(coder: Any, val: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    predictions = _predict_frame(coder, val)
    candidates = [json.loads(value) for value in predictions["candidates_json"]]
    return (
        predictions,
        float(predictions["correct"].mean()),
        accuracy_at_k(predictions["gold_code"], candidates, 5),
    )


def _metric_for_subgroup(table: pd.DataFrame, subgroup: str, metric: str) -> float | None:
    row = table[table["subgroup"] == subgroup]
    if row.empty or pd.isna(row.iloc[0][metric]):
        return None
    return float(row.iloc[0][metric])


def _select_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    chosen = frame.sort_values(
        ["val_accuracy_at_1", "val_accuracy_at_5", "complexity_rank", "history_weight"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return chosen.to_dict()


def run_advanced_singlelabel_benchmark(
    records: pd.DataFrame,
    terminology: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_auto_accuracy: float = 0.95,
    dense_model_name: str | None = None,
    cross_encoder_model_name: str | None = None,
    device: str | None = None,
    external_human_reference: bool = True,
    data_is_synthetic: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Validation-only model selection with an untouched held-out TEST split.

    Selection is staged to keep compute bounded:
    1. select lexical historical-memory weight;
    2. optionally select dense fusion weight;
    3. optionally select cross-encoder fusion weight.

    TEST is evaluated only after the final configuration is frozen.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = records.copy()
    if "split" not in data or not {"train", "val", "test"}.issubset(set(data["split"].astype(str))):
        data = assign_document_splits(data, seed=seed)
    assert_document_disjoint(data)
    train = data[data["split"] == "train"].copy()
    val = data[data["split"] == "val"].copy()
    test = data[data["split"] == "test"].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("train/val/test must all be non-empty")

    selection_rows: list[dict[str, Any]] = []
    lexical_scores: list[dict[str, Any]] = []
    for history_weight in [0.0, 0.25, 0.50, 0.75, 1.0]:
        coder = HistoricalCoder(history_weight=history_weight, top_k=10).fit(train, terminology)
        _, acc1, acc5 = _score_validation(coder, val)
        row = {
            "stage": "lexical_history",
            "history_weight": history_weight,
            "dense_weight": 0.0,
            "reranker_weight": 0.0,
            "val_accuracy_at_1": acc1,
            "val_accuracy_at_5": acc5,
            "complexity_rank": 0,
        }
        lexical_scores.append(row)
        selection_rows.append(row)
    best_lexical = _select_row(lexical_scores)
    selected_config = AdvancedModelConfig(history_weight=float(best_lexical["history_weight"]))

    dense_index = None
    if dense_model_name:
        dense_index = DenseSemanticIndex(dense_model_name, device=device).fit(train, terminology)
        dense_rows = [dict(best_lexical)]
        for dense_weight in [0.25, 0.50, 0.75, 1.0]:
            config = AdvancedModelConfig(
                history_weight=float(best_lexical["history_weight"]),
                dense_weight=dense_weight,
            )
            coder = AdvancedSingleLabelCoder(config, dense_index=dense_index).fit(train, terminology)
            _, acc1, acc5 = _score_validation(coder, val)
            row = {
                "stage": "dense_fusion",
                "history_weight": config.history_weight,
                "dense_weight": config.dense_weight,
                "reranker_weight": 0.0,
                "val_accuracy_at_1": acc1,
                "val_accuracy_at_5": acc5,
                "complexity_rank": 1,
            }
            dense_rows.append(row)
            selection_rows.append(row)
        best_dense = _select_row(dense_rows)
        selected_config = AdvancedModelConfig(
            history_weight=float(best_dense["history_weight"]),
            dense_weight=float(best_dense["dense_weight"]),
        )

    reranker = None
    if cross_encoder_model_name:
        reranker = CrossEncoderCandidateReranker(cross_encoder_model_name, device=device)
        rerank_rows = [{
            "stage": "cross_encoder_base",
            "history_weight": selected_config.history_weight,
            "dense_weight": selected_config.dense_weight,
            "reranker_weight": 0.0,
            "val_accuracy_at_1": None,
            "val_accuracy_at_5": None,
            "complexity_rank": 1 if selected_config.dense_weight > 0 else 0,
        }]
        base_coder = AdvancedSingleLabelCoder(selected_config, dense_index=dense_index).fit(train, terminology)
        _, base_acc1, base_acc5 = _score_validation(base_coder, val)
        rerank_rows[0]["val_accuracy_at_1"] = base_acc1
        rerank_rows[0]["val_accuracy_at_5"] = base_acc5
        for reranker_weight in [0.25, 0.50, 0.75]:
            config = AdvancedModelConfig(
                history_weight=selected_config.history_weight,
                dense_weight=selected_config.dense_weight,
                reranker_weight=reranker_weight,
            )
            coder = AdvancedSingleLabelCoder(
                config,
                dense_index=dense_index,
                reranker=reranker,
            ).fit(train, terminology)
            _, acc1, acc5 = _score_validation(coder, val)
            row = {
                "stage": "cross_encoder_fusion",
                "history_weight": config.history_weight,
                "dense_weight": config.dense_weight,
                "reranker_weight": config.reranker_weight,
                "val_accuracy_at_1": acc1,
                "val_accuracy_at_5": acc5,
                "complexity_rank": 2,
            }
            rerank_rows.append(row)
            selection_rows.append(row)
        best_rerank = _select_row(rerank_rows)
        selected_config = AdvancedModelConfig(
            history_weight=float(best_rerank["history_weight"]),
            dense_weight=float(best_rerank["dense_weight"]),
            reranker_weight=float(best_rerank["reranker_weight"]),
        )

    model_selection = pd.DataFrame(selection_rows)
    model_selection.to_csv(output / "model_selection.csv", index=False)

    selected_coder = AdvancedSingleLabelCoder(
        selected_config,
        dense_index=dense_index,
        reranker=reranker if selected_config.reranker_weight > 0 else None,
    ).fit(train, terminology)
    validation_predictions = _predict_frame(selected_coder, val)
    test_predictions = _predict_frame(selected_coder, test)
    baseline_coder = HistoricalCoder(history_weight=0.0, top_k=10).fit(train, terminology)
    baseline_test = _predict_frame(baseline_coder, test)

    threshold = choose_threshold_max_coverage(validation_predictions, target_auto_accuracy)
    test_predictions["decision"] = "HUMAN_REVIEW"
    if threshold is not None:
        test_predictions.loc[test_predictions["confidence"] >= threshold, "decision"] = "AUTO_CANDIDATE"

    seen_codes = {str(code) for code in train["gold_code"] if str(code)}
    diagnostics = annotate_prediction_diagnostics(test_predictions, seen_codes, candidate_k=10)
    candidates = [json.loads(value) for value in diagnostics["candidates_json"]]
    open_set = subgroup_metrics(diagnostics)
    coverage_curve = coverage_accuracy_curve(diagnostics)
    stress = policy_stress_test(validation_predictions, diagnostics)
    failures = failure_summary(diagnostics)
    auto = diagnostics["decision"] == "AUTO_CANDIDATE"

    selected_acc1 = float(diagnostics["correct"].mean())
    baseline_acc1 = float(baseline_test["correct"].mean())
    metrics = {
        "n_test": int(len(diagnostics)),
        "accuracy_at_1": selected_acc1,
        "accuracy_at_5": accuracy_at_k(diagnostics["gold_code"], candidates, 5),
        "candidate_recall_at_10": _metric_for_subgroup(open_set, "all", "candidate_recall_at_10"),
        "seen_code_accuracy_at_1": _metric_for_subgroup(open_set, "seen_code", "accuracy_at_1"),
        "unseen_code_accuracy_at_1": _metric_for_subgroup(open_set, "unseen_code", "accuracy_at_1"),
        "selected_history_weight": selected_config.history_weight,
        "selected_dense_weight": selected_config.dense_weight,
        "selected_reranker_weight": selected_config.reranker_weight,
        "selected_threshold": threshold,
        "target_auto_accuracy": target_auto_accuracy,
        "auto_candidate_rate": float(auto.mean()),
        "auto_candidate_accuracy": float(diagnostics.loc[auto, "correct"].mean()) if auto.any() else None,
        "human_review_rate": float((~auto).mean()),
        "terminology_only_accuracy_at_1": baseline_acc1,
        "advanced_accuracy_delta": selected_acc1 - baseline_acc1,
    }

    overlap = set(train["record_id"]) & set(test["record_id"])
    contract = contract_from_benchmark_metadata(
        external_human_reference=external_human_reference,
        group_disjoint_test=not overlap,
        candidate_dictionary_source="external",
        test_used_for_selection_or_tuning=False,
        data_is_synthetic=data_is_synthetic,
        provenance_recorded=True,
    )
    metrics["results_status"] = contract.status
    metrics["results_reportable"] = contract.reportable

    diagnostics.to_csv(output / "predictions.csv", index=False)
    validation_predictions.to_csv(output / "validation_predictions.csv", index=False)
    baseline_test.to_csv(output / "terminology_only_test_predictions.csv", index=False)
    open_set.to_csv(output / "open_set_metrics.csv", index=False)
    coverage_curve.to_csv(output / "coverage_accuracy.csv", index=False)
    stress.to_csv(output / "policy_stress_test.csv", index=False)
    failures.to_csv(output / "failure_summary.csv", index=False)
    diagnostics[diagnostics["correct"] == 0].to_csv(output / "error_analysis.csv", index=False)

    _dump(output / "metrics.json", metrics)
    _dump(output / "historical_memory_value.json", {
        "terminology_only_test_accuracy_at_1": baseline_acc1,
        "selected_advanced_test_accuracy_at_1": selected_acc1,
        "accuracy_at_1_delta": selected_acc1 - baseline_acc1,
        "interpretation": "Held-out descriptive comparison; all configuration selection used validation only.",
    })
    write_results_contract(output / "results_contract.json", contract)
    _dump(output / "leakage_audit.json", {"record_id_overlap": len(overlap)})
    _dump(output / "frozen_policy.json", {
        "version": "0.0.13",
        "model_type": "advanced_singlelabel",
        "history_weight": selected_config.history_weight,
        "dense_weight": selected_config.dense_weight,
        "reranker_weight": selected_config.reranker_weight,
        "dense_model_name": dense_model_name,
        "cross_encoder_model_name": cross_encoder_model_name,
        "decision_threshold": threshold,
        "threshold_selection_rule": "max_validation_coverage_subject_to_target_accuracy",
        "target_auto_accuracy": target_auto_accuracy,
    })
    _dump(output / "experiment_manifest.json", {
        "version": "0.0.13",
        "seed": seed,
        "selection_protocol": "validation_only_staged_model_selection",
        "test_used_for_selection_or_tuning": False,
        "external_human_reference": external_human_reference,
        "data_is_synthetic": data_is_synthetic,
    })
    _dump(output / "model_provenance.json", {
        "lexical_backend": "character_tfidf",
        "dense_model_name": dense_model_name,
        "cross_encoder_model_name": cross_encoder_model_name,
        "dense_enabled": bool(dense_model_name),
        "cross_encoder_enabled": bool(cross_encoder_model_name),
        "model_loading_note": "Optional external model names/paths are user supplied; weights are not bundled in this repository.",
    })
    _dump(output / "data_fingerprints.json", {
        "records_sha256": sha256(data.to_csv(index=False).encode()).hexdigest(),
        "terminology_sha256": sha256(terminology.to_csv(index=False).encode()).hexdigest(),
    })
    return metrics
