#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.knowledge import prepare_terminology_knowledge
from cohortcoder.mimic_benchmark import run_mimic_icd10_benchmark


def main() -> None:
    output = Path("outputs/multilabel_synthetic_demo")
    records = pd.DataFrame([
        {"record_id": "tr1", "subject_id": "s1", "text": "NSTEMI myocardial infarction and coronary atherosclerosis", "gold_codes_json": json.dumps(["I214", "I251"]), "split": "train"},
        {"record_id": "tr2", "subject_id": "s2", "text": "pneumonia treated with antibiotics", "gold_codes_json": json.dumps(["J189"]), "split": "train"},
        {"record_id": "tr3", "subject_id": "s3", "text": "heart failure with dyspnea", "gold_codes_json": json.dumps(["I509"]), "split": "train"},
        {"record_id": "tr4", "subject_id": "s4", "text": "coronary disease after myocardial infarction", "gold_codes_json": json.dumps(["I251", "I214"]), "split": "train"},
        {"record_id": "v1", "subject_id": "s5", "text": "non ST elevation myocardial infarction", "gold_codes_json": json.dumps(["I214"]), "split": "val"},
        {"record_id": "v2", "subject_id": "s6", "text": "pneumonia and heart failure", "gold_codes_json": json.dumps(["J189", "I509"]), "split": "val"},
        {"record_id": "te1", "subject_id": "s7", "text": "NSTEMI with coronary atherosclerotic disease", "gold_codes_json": json.dumps(["I214", "I251"]), "split": "test"},
        {"record_id": "te2", "subject_id": "s8", "text": "community pneumonia with chronic heart failure", "gold_codes_json": json.dumps(["J189", "I509"]), "split": "test"},
    ])
    terminology = prepare_terminology_knowledge(pd.DataFrame([
        {"system": "ICD-10", "code": "I214", "term": "Non-ST elevation myocardial infarction", "synonyms": "NSTEMI|myocardial infarction"},
        {"system": "ICD-10", "code": "I251", "term": "Atherosclerotic heart disease", "synonyms": "coronary atherosclerosis|coronary disease"},
        {"system": "ICD-10", "code": "J189", "term": "Pneumonia, unspecified organism", "synonyms": "pneumonia"},
        {"system": "ICD-10", "code": "I509", "term": "Heart failure, unspecified", "synonyms": "heart failure"},
    ]), coding_system="ICD-10")
    metrics = run_mimic_icd10_benchmark(
        records,
        terminology,
        output,
        target_proposal_precision=0.80,
        external_human_reference=False,
        data_is_synthetic=True,
        source_version="synthetic-demo",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
