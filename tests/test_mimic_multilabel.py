import json

import pandas as pd

from cohortcoder.mimic import prepare_mimic_iv_icd10, assert_subject_disjoint
from cohortcoder.multilabel import (
    MultiLabelHistoricalCoder,
    rank_dataframe,
    ranking_metrics,
    select_threshold_max_recall_at_precision,
    threshold_metrics,
)


def test_prepare_mimic_filters_icd10_and_splits_by_subject(tmp_path):
    discharge = pd.DataFrame([
        {"subject_id": "s1", "hadm_id": "h1", "note_id": "n1", "text": "acute myocardial infarction treated"},
        {"subject_id": "s1", "hadm_id": "h2", "note_id": "n2", "text": "heart failure follow up"},
        {"subject_id": "s2", "hadm_id": "h3", "note_id": "n3", "text": "pneumonia treated with antibiotics"},
    ])
    diagnoses = pd.DataFrame([
        {"subject_id": "s1", "hadm_id": "h1", "icd_code": "I214", "icd_version": "10"},
        {"subject_id": "s1", "hadm_id": "h1", "icd_code": "4109", "icd_version": "9"},
        {"subject_id": "s1", "hadm_id": "h2", "icd_code": "I509", "icd_version": "10"},
        {"subject_id": "s2", "hadm_id": "h3", "icd_code": "J189", "icd_version": "10"},
    ])
    dictionary = pd.DataFrame([
        {"icd_code": "I214", "icd_version": "10", "long_title": "Non-ST elevation myocardial infarction"},
        {"icd_code": "I509", "icd_version": "10", "long_title": "Heart failure, unspecified"},
        {"icd_code": "J189", "icd_version": "10", "long_title": "Pneumonia, unspecified organism"},
        {"icd_code": "4109", "icd_version": "9", "long_title": "Old ICD-9 label"},
    ])
    paths = []
    for name, frame in [("discharge.csv", discharge), ("diagnoses.csv", diagnoses), ("dictionary.csv", dictionary)]:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        paths.append(path)
    output = tmp_path / "prepared.csv"
    result = prepare_mimic_iv_icd10(*paths, output, seed=7)
    assert len(result) == 3
    assert all("4109" not in json.loads(value) for value in result.gold_codes_json)
    assert_subject_disjoint(result)
    # Both hospitalizations for s1 must share one split.
    assert result[result.subject_id == "s1"].split.nunique() == 1


def test_multilabel_coder_ranks_multiple_gold_codes():
    history = pd.DataFrame([
        {"record_id": "a", "text": "myocardial infarction and coronary disease", "gold_codes_json": json.dumps(["I214", "I251"])},
        {"record_id": "b", "text": "pneumonia infection", "gold_codes_json": json.dumps(["J189"])},
    ])
    terminology = pd.DataFrame([
        {"code": "I214", "term": "Non-ST elevation myocardial infarction"},
        {"code": "I251", "term": "Atherosclerotic heart disease"},
        {"code": "J189", "term": "Pneumonia unspecified organism"},
    ])
    test = pd.DataFrame([
        {"record_id": "x", "text": "non st elevation myocardial infarction with coronary disease", "gold_codes_json": json.dumps(["I214", "I251"])},
    ])
    coder = MultiLabelHistoricalCoder(history_weight=0.25, top_k=3).fit(history, terminology)
    ranked = rank_dataframe(coder, test)
    metrics = ranking_metrics(ranked, k_values=(2,))
    assert metrics["recall_at_2"] == 1.0


def test_threshold_selection_targets_code_proposal_precision():
    predictions = pd.DataFrame([
        {
            "record_id": "a",
            "gold_codes_json": json.dumps(["A", "B"]),
            "candidates_json": json.dumps([
                {"code": "A", "score": 0.9},
                {"code": "X", "score": 0.8},
                {"code": "B", "score": 0.7},
            ]),
        },
        {
            "record_id": "b",
            "gold_codes_json": json.dumps(["C"]),
            "candidates_json": json.dumps([
                {"code": "C", "score": 0.85},
                {"code": "Y", "score": 0.6},
            ]),
        },
    ])
    threshold = select_threshold_max_recall_at_precision(predictions, target_precision=0.95)
    assert threshold == 0.85
    metrics = threshold_metrics(predictions, threshold)
    assert metrics["micro_precision"] == 1.0
    assert metrics["n_code_proposals"] == 2
