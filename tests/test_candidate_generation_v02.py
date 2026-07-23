import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.candidate_generation import AliasAwareHybridCoder


def _history():
    return pd.DataFrame(
        [
            {"text": "head pain", "mention": "head pain", "gold_code": "A", "gold_term": "Headache"},
            {"text": "weak muscles", "mention": "weak muscles", "gold_code": "B", "gold_term": "Muscle weakness"},
        ]
    )


def _terminology():
    return pd.DataFrame(
        [
            {"code": "A", "term": "Headache", "synonyms": "cephalalgia | head pain"},
            {"code": "B", "term": "Muscle weakness", "synonyms": "weak muscles"},
            {"code": "C", "term": "Nausea", "synonyms": "naseua | nuasea | feeling sick"},
        ]
    )


def test_alias_only_candidate_can_reach_top1():
    coder = AliasAwareHybridCoder(history_weight=0.0, word_weight=0.2, top_k=3).fit(
        _history(), _terminology()
    )
    pred = coder.predict_one("naseua")
    assert pred.code == "C"
    assert pred.candidates[0]["matched_alias"].lower() == "naseua"


def test_candidate_rows_expose_auditable_component_scores():
    coder = AliasAwareHybridCoder(history_weight=0.25, word_weight=0.35, top_k=3).fit(
        _history(), _terminology()
    )
    pred = coder.predict_one("head pain")
    candidate = pred.candidates[0]
    assert candidate["code"] == "A"
    assert set(["matched_alias", "char_score", "word_score", "history_score", "terminology_score"]).issubset(candidate)
    assert candidate["score"] >= pred.candidates[1]["score"]


def test_invalid_weights_are_rejected():
    try:
        AliasAwareHybridCoder(history_weight=1.2)
    except ValueError:
        pass
    else:
        raise AssertionError("history_weight > 1 should fail")

    try:
        AliasAwareHybridCoder(word_weight=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("word_weight < 0 should fail")
