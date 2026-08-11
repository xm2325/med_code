#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cohortcoder.mipa_local_inference import InferenceConfig, generate_local_predictions, parse_command
from cohortcoder.mipa_phenotyping import DEFAULT_PHENOTYPES


def _local_path(value: str) -> str:
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "s3://", "gs://")):
        raise argparse.ArgumentTypeError("Restricted clinical inputs and outputs must use local filesystem paths.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate evidence-grounded MIPA phenotype predictions using a user-supplied local command backend. "
            "The backend receives one JSON request on stdin and must return one strict JSON object on stdout."
        )
    )
    parser.add_argument("--notes", required=True, type=_local_path, help="Local authorised golden_notes.csv")
    parser.add_argument("--predictions", required=True, type=_local_path, help="Checkpointed local JSONL output")
    parser.add_argument("--failures", required=True, type=_local_path, help="Local JSONL failure log")
    parser.add_argument(
        "--backend-command",
        required=True,
        help=(
            "Local executable command, for example 'python /secure/local_backend.py'. "
            "No shell is used. The approved environment must enforce network isolation for the child process."
        ),
    )
    parser.add_argument("--model-id", default="local-model")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--phenotypes", nargs="+", default=list(DEFAULT_PHENOTYPES))
    parser.add_argument("--limit-notes", type=int)
    parser.add_argument("--no-resume", action="store_true", help="Overwrite prediction/failure files and start again")
    parser.add_argument("--summary", type=_local_path, help="Optional local JSON summary path")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless every requested note/phenotype pair has a valid parsed prediction.",
    )
    args = parser.parse_args()

    config = InferenceConfig(
        command=parse_command(args.backend_command),
        timeout_seconds=args.timeout_seconds,
        model_id=args.model_id,
        resume=not args.no_resume,
    )
    summary = generate_local_predictions(
        notes_path=args.notes,
        predictions_path=args.predictions,
        failures_path=args.failures,
        config=config,
        phenotypes=args.phenotypes,
        limit_notes=args.limit_notes,
    )

    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")

    if args.require_complete and not summary["complete"]:
        raise SystemExit("Local inference generation incomplete; inspect the failure log before evaluation.")


if __name__ == "__main__":
    main()
