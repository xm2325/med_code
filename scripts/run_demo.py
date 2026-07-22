#!/usr/bin/env python
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder import HistoricalCoder, accuracy_at_k, coverage_accuracy


def main() -> None:
    history = pd.DataFrame([
        {"text": "severe aching muscles in both legs", "gold_code": "M1", "gold_term": "Myalgia"},
        {"text": "my legs became very weak", "gold_code": "M2", "gold_term": "Muscular weakness"},
        {"text": "felt sick and nauseated every morning", "gold_code": "G1", "gold_term": "Nausea"},
    ])
    terminology = pd.DataFrame([
        {"code": "M1", "term": "Myalgia muscle pain aching muscles"},
        {"code": "M2", "term": "Muscular weakness weak muscles weakness"},
        {"code": "G1", "term": "Nausea feeling sick nauseated"},
        {"code": "R1", "term": "Rash skin eruption red itchy rash"},
    ])
    test = pd.DataFrame([
        {"text": "persistent aching muscle pain in my thighs", "gold_code": "M1"},
        {"text": "I feel nauseated after taking the tablets", "gold_code": "G1"},
        {"text": "a new red itchy eruption appeared", "gold_code": "R1"},
    ])

    coder = HistoricalCoder(history_weight=0.25, top_k=4).fit(history, terminology)
    predictions = coder.predict(test["text"])
    print(f"Accuracy@1: {accuracy_at_k(test.gold_code, [p.candidates for p in predictions], 1):.3f}")
    print(f"Accuracy@3: {accuracy_at_k(test.gold_code, [p.candidates for p in predictions], 3):.3f}")
    print("Selective policy example:", coverage_accuracy(test.gold_code, predictions, threshold=0.15))
    for text, pred in zip(test.text, predictions):
        print(f"\n{text}\n -> {pred.code} | {pred.term} | confidence={pred.confidence:.3f}")


if __name__ == "__main__":
    main()
