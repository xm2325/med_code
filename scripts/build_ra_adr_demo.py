#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from cohortcoder.adr_coding import (
    DEMO_TERMINOLOGY_SOURCE,
    RA_DRUG_ALIASES,
    OneLineADRCoder,
    demo_terminology,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "ra_adr_synthetic_50.json"
TEMPLATE = ROOT / "demo" / "ra_adr_meddra_demo.template.html"
OUTPUT = ROOT / "demo" / "ra_adr_meddra_demo.html"


def main() -> None:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(rows) != 50:
        raise ValueError(f"Expected exactly 50 synthetic records, found {len(rows)}")

    coder = OneLineADRCoder()
    records = []
    for row in rows:
        prediction = coder.map_line(row["text"], record_id=row["record_id"])
        records.append({**row, **prediction})

    payload = {
        "records": records,
        "terms": demo_terminology(),
        "drugAliases": RA_DRUG_ALIASES,
        "terminology": DEMO_TERMINOLOGY_SOURCE,
        "method": {
            "active": "evidence_first_baseline",
            "baseline_executed_in_demo": True,
            "prompt_only_executed_in_demo": False,
            "rag_executed_in_demo": False,
            "llm_code_generation_allowed": False,
        },
        "generatedFrom": str(SOURCE.relative_to(ROOT)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__MEDCODE_PAYLOAD__", encoded)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Built {OUTPUT.relative_to(ROOT)} with {len(records)} synthetic records")


if __name__ == "__main__":
    main()
