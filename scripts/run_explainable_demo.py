#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.core import HistoricalCoder
from cohortcoder.explain import explain_predictions, write_explanation_artifacts
from cohortcoder.knowledge import attach_knowledge_provenance, prepare_terminology_knowledge


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fully synthetic explainable coding demo")
    parser.add_argument("--output-dir", default="outputs/explainable_demo")
    args = parser.parse_args()

    history = pd.DataFrame([
        {"text": "I developed aching muscles in both legs", "mention": "aching muscles", "gold_code": "DEMO-M1", "gold_term": "Myalgia"},
        {"text": "My legs felt weak when walking", "mention": "legs felt weak", "gold_code": "DEMO-M2", "gold_term": "Muscular weakness"},
        {"text": "I felt sick after taking the tablets", "mention": "felt sick", "gold_code": "DEMO-G1", "gold_term": "Nausea"},
    ])
    terminology = prepare_terminology_knowledge(pd.DataFrame([
        {"system": "DEMO-MEDDRA", "code": "DEMO-M1", "term": "Myalgia", "synonyms": "muscle pain|aching muscles", "definition": "Synthetic demo definition for muscle pain.", "knowledge_source": "synthetic-demo"},
        {"system": "DEMO-MEDDRA", "code": "DEMO-M2", "term": "Muscular weakness", "synonyms": "weak muscles|muscle weakness", "definition": "Synthetic demo definition for muscular weakness.", "knowledge_source": "synthetic-demo"},
        {"system": "DEMO-MEDDRA", "code": "DEMO-G1", "term": "Nausea", "synonyms": "feeling sick|felt sick", "definition": "Synthetic demo definition for nausea.", "knowledge_source": "synthetic-demo"},
    ]))
    record = pd.DataFrame([{
        "record_id": "SYNTHETIC-NEW-001",
        "text": "At follow-up, the patient reported persistent severe muscle pain in both thighs after treatment.",
        "mention": "severe muscle pain",
    }])

    coder = HistoricalCoder(history_weight=0.25, top_k=3).fit(history, terminology)
    prediction = coder.predict_one(record.iloc[0].mention)
    predictions = pd.DataFrame([{
        "record_id": record.iloc[0].record_id,
        "text": record.iloc[0].text,
        "mention": record.iloc[0].mention,
        "predicted_code": prediction.code,
        "predicted_term": prediction.term,
        "confidence": prediction.confidence,
        "decision": "DEMO_ONLY",
        "candidates_json": json.dumps(prediction.candidates),
        "historical_cases_json": json.dumps(prediction.historical_cases),
    }])
    explanations = explain_predictions(predictions, terminology, coder=coder)
    explanations = attach_knowledge_provenance(explanations, terminology)
    metrics = write_explanation_artifacts(Path(args.output_dir), explanations)
    print(json.dumps({
        "prediction": {"code": prediction.code, "term": prediction.term, "confidence": prediction.confidence},
        "why": explanations[0]["why"],
        "evidence": explanations[0]["evidence_quotes"],
        "knowledge_source": explanations[0]["external_knowledge"]["knowledge_source"],
        "metrics": metrics,
        "html": str(Path(args.output_dir) / "explanations.html"),
    }, indent=2))


if __name__ == "__main__":
    main()
