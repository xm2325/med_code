import json

import pandas as pd

from cohortcoder.multilabel import MultiLabelHistoricalCoder, rank_dataframe
from cohortcoder.multilabel_batch import rank_dataframe_batched


def test_batched_ranking_matches_single_record_ranking():
    history = pd.DataFrame([
        {"record_id": "a", "text": "myocardial infarction coronary disease", "gold_codes_json": json.dumps(["I214", "I251"])},
        {"record_id": "b", "text": "pneumonia infection", "gold_codes_json": json.dumps(["J189"])},
        {"record_id": "c", "text": "heart failure edema", "gold_codes_json": json.dumps(["I509"])},
    ])
    terminology = pd.DataFrame([
        {"code": "I214", "term": "Non-ST elevation myocardial infarction"},
        {"code": "I251", "term": "Atherosclerotic heart disease"},
        {"code": "J189", "term": "Pneumonia unspecified organism"},
        {"code": "I509", "term": "Heart failure unspecified"},
    ])
    records = pd.DataFrame([
        {"record_id": "x", "text": "myocardial infarction with coronary disease", "gold_codes_json": json.dumps(["I214", "I251"])},
        {"record_id": "y", "text": "pneumonia with heart failure", "gold_codes_json": json.dumps(["J189", "I509"])},
    ])
    coder = MultiLabelHistoricalCoder(history_weight=0.25, top_k=4).fit(history, terminology)
    single = rank_dataframe(coder, records)
    batched = rank_dataframe_batched(coder, records, batch_size=1)
    for left, right in zip(single.candidates_json, batched.candidates_json):
        left_codes = [item["code"] for item in json.loads(left)]
        right_codes = [item["code"] for item in json.loads(right)]
        assert left_codes == right_codes
