#!/usr/bin/env python
"""Accelerated entry point for the v0.2 candidate-generation benchmark.

The benchmark design, split, model-selection rules, metrics, and output contract remain in
``run_candidate_generation_v02.py``. This wrapper changes execution only:

1. predictions are made in batches so sparse query transforms/similarity matrices are reused;
2. the six validation weight configurations share one fitted alias/history retrieval state,
   because those weights affect score fusion but not the fitted vectorizers/matrices.

No TEST labels or TEST metrics are used to select configuration.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_candidate_generation_v02 as benchmark


class CachedFitAliasCoder(benchmark.AliasAwareHybridCoder):
    """Reuse immutable fitted retrieval state within this one benchmark process."""

    _fit_cache: dict[tuple[int, int], dict] = {}

    def fit(self, history: pd.DataFrame, terminology: pd.DataFrame):
        key = (id(history), id(terminology))
        cached = self._fit_cache.get(key)
        if cached is None:
            super().fit(history, terminology)
            cached = {name: value for name, value in self.__dict__.items() if name != "config"}
            self._fit_cache[key] = cached
        else:
            self.__dict__.update(cached)
        return self


def batch_predict_rows(coder, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[list[dict]]]:
    mentions = frame["mention"].astype(str).tolist()
    predictions = coder.predict(mentions)
    rows: list[dict] = []
    candidate_lists: list[list[dict]] = []
    for (_, record), prediction in zip(frame.iterrows(), predictions):
        candidates = prediction.candidates
        candidate_lists.append(candidates)
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "source_dataset": str(record["source_dataset"]),
                "phrase": str(record["mention"]),
                "gold_code": str(record["gold_code"]),
                "predicted_code": str(prediction.code),
                "confidence": float(prediction.confidence),
                "correct": int(str(prediction.code) == str(record["gold_code"])),
                "candidates_json": json.dumps(candidates, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows), candidate_lists


benchmark.AliasAwareHybridCoder = CachedFitAliasCoder
benchmark.predict_rows = batch_predict_rows

if __name__ == "__main__":
    benchmark.main()
