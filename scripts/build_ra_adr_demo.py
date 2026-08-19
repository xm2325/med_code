#!/usr/bin/env python
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "ra_adr_synthetic_50.json"
TEMPLATE = ROOT / "demo" / "ra_adr_meddra_demo.template.html"
OUTPUT = ROOT / "demo" / "ra_adr_meddra_demo.html"


def load_demo_coder_module():
    """Load the lightweight demo coder without importing optional ML dependencies."""
    package = types.ModuleType("cohortcoder")
    package.__path__ = [str(ROOT / "src" / "cohortcoder")]
    sys.modules.setdefault("cohortcoder", package)
    for name in ("clinical_context", "adr_coding"):
        qualified = f"cohortcoder.{name}"
        spec = importlib.util.spec_from_file_location(qualified, ROOT / "src" / "cohortcoder" / f"{name}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {qualified}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return sys.modules["cohortcoder.adr_coding"]


def main() -> None:
    adr = load_demo_coder_module()
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(rows) != 50:
        raise ValueError(f"Expected exactly 50 synthetic records, found {len(rows)}")

    coder = adr.OneLineADRCoder()
    records = []
    for row in rows:
        prediction = coder.map_line(row["text"], record_id=row["record_id"])
        records.append({**row, **prediction})

    payload = {
        "records": records,
        "terms": adr.demo_terminology(),
        "drugAliases": adr.RA_DRUG_ALIASES,
        "aliasSource": "local_demo_curated",
        "terminology": adr.DEMO_TERMINOLOGY_SOURCE,
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
