import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder import HistoricalCoder, accuracy_at_k, coverage_accuracy


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
    coder = HistoricalCoder(history_weight=0.25, top_k=3).fit(history, terminology)
    pred = coder.predict_one("aching muscle pain")
    assert pred.code == "A"
    assert len(pred.candidates) == 3
    assert pred.candidates[0]["score"] >= pred.candidates[1]["score"]


def test_unseen_historical_code_can_be_retrieved_from_terminology():
    history, terminology = fixtures()
    coder = HistoricalCoder(history_weight=0.25).fit(history, terminology)
    pred = coder.predict_one("new itchy red skin eruption")
    assert pred.code == "C"


def test_accuracy_at_k():
    candidates = [[{"code": "A"}, {"code": "B"}], [{"code": "A"}, {"code": "C"}]]
    assert accuracy_at_k(["A", "C"], candidates, 1) == 0.5
    assert accuracy_at_k(["A", "C"], candidates, 2) == 1.0


def test_selective_policy_output():
    history, terminology = fixtures()
    coder = HistoricalCoder(history_weight=0.25).fit(history, terminology)
    preds = coder.predict(["aching muscle pain", "itchy red skin eruption"])
    result = coverage_accuracy(["A", "C"], preds, threshold=0.0)
    assert result["coverage"] == 1.0
    assert result["accuracy"] == 1.0
