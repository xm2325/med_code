import json

import pandas as pd

from cohortcoder.multilabel import MultiLabelHistoricalCoder, rank_dataframe
from cohortcoder.multilabel_explain import explain_multilabel_proposals


def test_multilabel_explanations_are_per_record_code_pair():
    history = pd.DataFrame([
        {"record_id": "h1", "text": "pneumonia treated", "gold_codes_json": json.dumps(["J189"])},
        {"record_id": "h2", "text": "heart failure treated", "gold_codes_json": json.dumps(["I509"])},
    ])
    terminology = pd.DataFrame([
        {"system": "ICD-10", "code": "J189", "term": "Pneumonia unspecified organism", "synonyms": "pneumonia"},
        {"system": "ICD-10", "code": "I509", "term": "Heart failure unspecified", "synonyms": "heart failure"},
    ])
    records = pd.DataFrame([
        {"record_id": "r1", "text": "Pneumonia improved. Heart failure remained stable.", "gold_codes_json": json.dumps(["J189", "I509"])},
    ])
    coder = MultiLabelHistoricalCoder(history_weight=0.25, top_k=2).fit(history, terminology)
    predictions = rank_dataframe(coder, records)
    explanations = explain_multilabel_proposals(
        records,
        predictions,
        terminology,
        coder,
        threshold=None,
        fallback_top_k=2,
    )
    assert len(explanations) == 2
    assert {item["predicted_code"] for item in explanations} == {"J189", "I509"}
    assert all(item["record_id"] == "r1" for item in explanations)
