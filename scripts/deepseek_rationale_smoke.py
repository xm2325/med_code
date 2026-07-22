#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.llm import DeepSeekRationaleClient


EXPLANATION = {
    "record_id": "SYNTHETIC-001",
    "coding_system": "DEMO",
    "predicted_code": "DEMO-A",
    "predicted_term": "Myalgia",
    "evidence_quotes": ["severe muscle pain"],
    "external_knowledge": {
        "term": "Myalgia",
        "synonyms": ["muscle pain"],
        "definition": "Synthetic demonstration knowledge only.",
        "hierarchy": "",
    },
}


if __name__ == "__main__":
    client = DeepSeekRationaleClient()
    result = client.generate(
        EXPLANATION,
        allow_external_llm=True,
        data_classification="synthetic",
    )
    # Never print the API key. This output is safe synthetic smoke-test metadata.
    print(json.dumps({
        "accepted": result["accepted"],
        "model": result["model"],
        "validation_errors": result["validation_errors"],
        "payload": result["payload"],
    }, indent=2))
    if not result["accepted"]:
        raise SystemExit(1)
