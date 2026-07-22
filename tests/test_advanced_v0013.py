import json

import pandas as pd

from cohortcoder.advanced import AdvancedModelConfig, AdvancedSingleLabelCoder
from cohortcoder.advanced_benchmark import run_advanced_singlelabel_benchmark


class FakeDense:
    is_fitted = True

    def score(self, text):
        return {
            "A": {"term_score": 0.1, "history_score": 0.1},
            "B": {"term_score": 0.9, "history_score": 0.9},
        }


class FakeReranker:
    def score_candidates(self, query, candidates):
        return {str(item["code"]): (1.0 if str(item["code"]) == "B" else 0.0) for item in candidates}


def _history():
    return pd.DataFrame([
        {"text": "alpha symptom", "mention": "alpha", "gold_code": "A", "gold_term": "Alpha"},
        {"text": "beta symptom", "mention": "beta", "gold_code": "B", "gold_term": "Beta"},
    ])


def _terminology():
    return pd.DataFrame([
        {"code": "A", "term": "Alpha", "search_text": "alpha"},
        {"code": "B", "term": "Beta", "search_text": "beta"},
    ])


def test_advanced_coder_keeps_fixed_code_dictionary():
    coder = AdvancedSingleLabelCoder(
        AdvancedModelConfig(history_weight=0.5, dense_weight=0.8, reranker_weight=0.5),
        dense_index=FakeDense(),
        reranker=FakeReranker(),
    ).fit(_history(), _terminology())
    prediction = coder.predict_one("beta")
    assert prediction.code in {"A", "B"}
    assert {item["code"] for item in prediction.candidates}.issubset({"A", "B"})
    assert prediction.code == "B"


def test_advanced_benchmark_freezes_validation_selected_policy(tmp_path):
    rows = []
    for split, values in {
        "train": [("t1", "alpha", "A"), ("t2", "beta", "B")],
        "val": [("v1", "alpha", "A"), ("v2", "beta", "B")],
        "test": [("x1", "alpha", "A"), ("x2", "beta", "B")],
    }.items():
        for record_id, mention, code in values:
            rows.append({
                "record_id": record_id,
                "text": mention,
                "mention": mention,
                "gold_code": code,
                "gold_term": "Alpha" if code == "A" else "Beta",
                "split": split,
            })
    metrics = run_advanced_singlelabel_benchmark(
        pd.DataFrame(rows),
        _terminology(),
        tmp_path,
        data_is_synthetic=True,
        external_human_reference=False,
    )
    policy = json.loads((tmp_path / "frozen_policy.json").read_text())
    manifest = json.loads((tmp_path / "experiment_manifest.json").read_text())
    assert policy["model_type"] == "advanced_singlelabel"
    assert manifest["test_used_for_selection_or_tuning"] is False
    assert metrics["results_reportable"] is False
    assert (tmp_path / "model_selection.csv").exists()
