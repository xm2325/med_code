import pandas as pd

from cohortcoder import HistoricalCoder


def test_terminology_only_scores_do_not_depend_on_historical_text_corpus():
    terminology = pd.DataFrame([
        {"code": "A", "term": "Myalgia muscle pain aching"},
        {"code": "B", "term": "Muscular weakness weak muscles"},
    ])
    history_one = pd.DataFrame([
        {"text": "completely unrelated historical vocabulary alpha beta", "gold_code": "A", "gold_term": "Myalgia"},
        {"text": "another unrelated historical phrase gamma delta", "gold_code": "B", "gold_term": "Muscular weakness"},
    ])
    history_two = pd.DataFrame([
        {"text": "muscle pain muscle pain muscle pain", "gold_code": "A", "gold_term": "Myalgia"},
        {"text": "weak muscles weak muscles weak muscles", "gold_code": "B", "gold_term": "Muscular weakness"},
    ])

    first = HistoricalCoder(history_weight=0.0, top_k=2).fit(history_one, terminology).predict_one("aching muscle pain")
    second = HistoricalCoder(history_weight=0.0, top_k=2).fit(history_two, terminology).predict_one("aching muscle pain")

    assert first.code == second.code == "A"
    assert [row["terminology_score"] for row in first.candidates] == [row["terminology_score"] for row in second.candidates]
    assert [row["score"] for row in first.candidates] == [row["score"] for row in second.candidates]
