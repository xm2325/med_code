import json

import pandas as pd

from cohortcoder.mimic_audit import audit_mimic_records


def _records():
    return pd.DataFrame([
        {"record_id": "a", "subject_id": "s1", "text": "Diagnosis one", "gold_codes_json": json.dumps(["A"]), "split": "train"},
        {"record_id": "b", "subject_id": "s2", "text": "Diagnosis two", "gold_codes_json": json.dumps(["B"]), "split": "val"},
        {"record_id": "c", "subject_id": "s3", "text": "Diagnosis three", "gold_codes_json": json.dumps(["C"]), "split": "test"},
    ])


def test_mimic_audit_ready_when_subjects_and_dictionary_are_clean():
    records = _records()
    terminology = pd.DataFrame([{"code": code, "term": code} for code in ["A", "B", "C"]])
    audit = audit_mimic_records(records, terminology)
    assert audit["ready_for_benchmark"]
    assert audit["terminology_coverage"] == 1.0
    assert not any(audit["subject_overlap"].values())


def test_mimic_audit_blocks_subject_leakage():
    records = _records()
    records.loc[2, "subject_id"] = "s1"
    terminology = pd.DataFrame([{"code": code, "term": code} for code in ["A", "B", "C"]])
    audit = audit_mimic_records(records, terminology)
    assert not audit["ready_for_benchmark"]
    assert "subject_id_leakage_across_splits" in audit["hard_failures"]


def test_mimic_audit_blocks_missing_terminology_codes():
    records = _records()
    terminology = pd.DataFrame([{"code": "A", "term": "A"}])
    audit = audit_mimic_records(records, terminology)
    assert not audit["ready_for_benchmark"]
    assert "terminology_coverage_below_threshold" in audit["hard_failures"]
