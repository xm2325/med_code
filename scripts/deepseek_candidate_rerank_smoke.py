#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.llm_rerank import DeepSeekCandidateReranker


def main() -> None:
    client = DeepSeekCandidateReranker()
    candidates = [
        {"code": "DEMO-A", "term": "Myalgia", "score": 0.55},
        {"code": "DEMO-B", "term": "Arthralgia", "score": 0.52},
    ]
    result = client.rerank(
        "severe muscle pain",
        candidates,
        allow_external_llm=True,
        data_classification="synthetic",
        evidence_quotes=["severe muscle pain"],
    )
    if not result["accepted"]:
        raise SystemExit("DeepSeek candidate rerank smoke failed: " + json.dumps(result))
    ranked = result["payload"]["ranked_codes"]
    if set(ranked) != {"DEMO-A", "DEMO-B"}:
        raise SystemExit("Candidate set changed unexpectedly")
    print(json.dumps({
        "accepted": True,
        "model": result["model"],
        "candidate_set_preserved": True,
        "top_code": ranked[0],
    }, indent=2))


if __name__ == "__main__":
    main()
