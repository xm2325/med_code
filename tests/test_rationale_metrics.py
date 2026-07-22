import json

import pandas as pd

from cohortcoder.rationale_metrics import evaluate_rationale_overlap, validate_rationale_offsets


def test_rationale_overlap_is_computed_per_record_code():
    predicted = pd.DataFrame([
        {
            "record_id": "r1",
            "predicted_code": "A",
            "evidence_spans_json": json.dumps([{"start": 10, "end": 20, "quote": "abcdefghij"}]),
        }
    ])
    reference = pd.DataFrame([
        {"record_id": "r1", "code": "A", "start": 15, "end": 25},
    ])
    summary, detail = evaluate_rationale_overlap(predicted, reference)
    assert summary["n_reference_record_code_pairs"] == 1
    assert detail.iloc[0].char_precision == 0.5
    assert detail.iloc[0].char_recall == 0.5
    assert detail.iloc[0].char_f1 == 0.5


def test_reference_offsets_must_match_source_quote():
    records = pd.DataFrame([{"record_id": "r1", "text": "patient has severe muscle pain today"}])
    start = records.iloc[0].text.index("severe muscle pain")
    reference = pd.DataFrame([
        {
            "record_id": "r1",
            "code": "A",
            "start": start,
            "end": start + len("severe muscle pain"),
            "quote": "severe muscle pain",
        }
    ])
    audit = validate_rationale_offsets(reference, records)
    assert audit["valid"] is True
