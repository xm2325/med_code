#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cohortcoder.scientific_acceptance import assess_current_scientific_evidence


def _load_optional(path: str | None):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a conservative scientific conclusion matrix for the RA comorbidity project. "
            "External benchmark evidence is kept separate from MedCode's own real-data evidence."
        )
    )
    parser.add_argument("--public-mipa-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--external-llm-macro-f1",
        type=float,
        default=0.891,
        help=(
            "Published fixed-configuration MIPA cross-model macro-F1 used only as external feasibility evidence; "
            "default 0.891 is the reported DeepSeek-R1 overall macro-F1, not a MedCode result."
        ),
    )
    parser.add_argument("--own-stage12-summary")
    parser.add_argument("--own-discordance-summary")
    args = parser.parse_args()

    public_summary = json.loads(Path(args.public_mipa_summary).read_text(encoding="utf-8"))
    result = assess_current_scientific_evidence(
        public_mipa_summary=public_summary,
        external_llm_macro_f1=args.external_llm_macro_f1,
        own_stage12_summary=_load_optional(args.own_stage12_summary),
        own_discordance_summary=_load_optional(args.own_discordance_summary),
    )
    result["external_evidence"] = {
        "source": "Yamga E, Murphy S, Despres P. A Systematic Exploration of LLM Behavior for EHR Phenotyping. medRxiv. 2026.",
        "doi": "10.64898/2026.04.16.26350890",
        "reported_deepseek_r1_macro_f1": args.external_llm_macro_f1,
        "interpretation": "External benchmark feasibility only; not MedCode performance.",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
