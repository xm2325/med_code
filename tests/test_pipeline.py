import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder import HistoricalCoder, accuracy_at_k, coverage_accuracy
from cohortcoder.analysis import (
    annotate_prediction_diagnostics,
    choose_threshold_max_coverage,
    policy_stress_test,
    subgroup_metrics,
)
from cohortcoder.results import build_results_contract, contract_from_benchmark_metadata
from cohortcoder.realdata import assign_document_splits, assert_document_disjoint


def fixtures():
    history = pd.DataFrame([
        {"text": "severe muscle aching", "gold_code": "A", "gold_term": "Myalgia"},
        {"text": "very weak muscles", "gold_code": "B", "gold_term": "Muscular weakness"},
    ])
    terminology = pd.DataFrame([
        {"code": "A", "term": "Myalgia muscle pain aching"},
        {"code": "B", "term": "Muscular weakness weak muscles"},
        {"code": "C", "term": "Rash red itchy skin eruption"},
    ])
    return history, terminology


def test_fit_predict_returns_ranked_candidates():
    history, terminology = fixtures()
    pred = HistoricalCoder(history_weight=0.25, top_k=3).fit(history, terminology).predict_one("aching muscle pain")
    assert pred.code == "A"
    assert len(pred.candidates) == 3
    assert pred.candidates[0]["score"] >= pred.candidates[1]["score"]


def test_unseen_historical_code_can_be_retrieved_from_terminology():
    history, terminology = fixtures()
    pred = HistoricalCoder(history_weight=0.25).fit(history, terminology).predict_one("new itchy red skin eruption")
    assert pred.code == "C"


def test_accuracy_at_k():
    candidates = [[{"code": "A"}, {"code": "B"}], [{"code": "A"}, {"code": "C"}]]
    assert accuracy_at_k(["A", "C"], candidates, 1) == 0.5
    assert accuracy_at_k(["A", "C"], candidates, 2) == 1.0


def test_selective_policy_output():
    history, terminology = fixtures()
    preds = HistoricalCoder(history_weight=0.25).fit(history, terminology).predict(["aching muscle pain", "itchy red skin eruption"])
    result = coverage_accuracy(["A", "C"], preds, threshold=0.0)
    assert result["coverage"] == 1.0
    assert result["accuracy"] == 1.0


def test_missing_results_metadata_is_not_reportable():
    contract = build_results_contract({})
    assert contract.reportable is False


def test_safe_real_benchmark_contract_is_reportable():
    contract = contract_from_benchmark_metadata(
        external_human_reference=True,
        group_disjoint_test=True,
        candidate_dictionary_source="external",
        test_used_for_selection_or_tuning=False,
        data_is_synthetic=False,
        provenance_recorded=True,
    )
    assert contract.reportable is True
    assert contract.status == "reportable"


def test_oracle_terminology_is_not_reportable():
    contract = contract_from_benchmark_metadata(
        external_human_reference=True,
        group_disjoint_test=True,
        candidate_dictionary_source="all_gold_oracle",
        test_used_for_selection_or_tuning=False,
        data_is_synthetic=False,
        provenance_recorded=True,
    )
    assert contract.reportable is False
    assert contract.status == "oracle_diagnostic"


def test_document_split_keeps_source_document_together():
    df = pd.DataFrame({"record_id": ["a", "a", "b", "c", "d", "e", "f", "g"], "text": ["x"] * 8, "mention": ["x"] * 8})
    out = assign_document_splits(df, seed=11, train=0.5, val=0.25)
    assert_document_disjoint(out)
    assert out.groupby("record_id")["split"].nunique().max() == 1


def test_threshold_selection_maximises_validation_coverage_subject_to_accuracy():
    validation = pd.DataFrame({
        "confidence": [0.90, 0.80, 0.70],
        "correct": [1, 1, 0],
    })
    # 0.90 reaches 100% accuracy at 1/3 coverage, but 0.80 reaches the same
    # target at 2/3 coverage and must therefore be selected.
    assert choose_threshold_max_coverage(validation, 1.0) == 0.80


def test_seen_unseen_and_failure_taxonomy():
    predictions = pd.DataFrame([
        {
            "gold_code": "A",
            "predicted_code": "A",
            "correct": 1,
            "candidates_json": json.dumps([{"code": "A"}, {"code": "B"}]),
        },
        {
            "gold_code": "C",
            "predicted_code": "B",
            "correct": 0,
            "candidates_json": json.dumps([{"code": "B"}, {"code": "C"}]),
        },
        {
            "gold_code": "D",
            "predicted_code": "B",
            "correct": 0,
            "candidates_json": json.dumps([{"code": "B"}, {"code": "C"}]),
        },
    ])
    diagnostics = annotate_prediction_diagnostics(predictions, {"A", "B", "C"})
    assert diagnostics.loc[0, "error_type"] == "correct"
    assert diagnostics.loc[1, "error_type"] == "ranking_failure"
    assert diagnostics.loc[2, "error_type"] == "candidate_generation_failure"
    assert diagnostics.loc[2, "code_novelty"] == "unseen_code"
    metrics = subgroup_metrics(diagnostics)
    unseen = metrics[metrics.subgroup == "unseen_code"].iloc[0]
    assert unseen.n == 1
    assert unseen.accuracy_at_1 == 0.0


def test_policy_stress_threshold_is_selected_from_validation_only():
    validation = pd.DataFrame({
        "confidence": [0.95, 0.85, 0.60],
        "correct": [1, 1, 0],
    })
    test = pd.DataFrame({
        "confidence": [0.90, 0.80, 0.70],
        "correct": [1, 0, 1],
    })
    result = policy_stress_test(validation, test, targets=[1.0])
    assert result.iloc[0].validation_threshold == 0.85
    assert result.iloc[0].test_n_auto == 1
    assert result.iloc[0].test_accuracy == 1.0
