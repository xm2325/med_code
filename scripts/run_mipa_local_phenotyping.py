#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cohortcoder.mipa_phenotyping import DEFAULT_PHENOTYPES, AcceptanceThresholds, write_evaluation_outputs
from cohortcoder.mipa_strict_acceptance import EvidenceAuditCoveragePolicy, evaluate_mipa_predictions_strict


def _local_path(value: str) -> str:
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "s3://", "gs://")):
        raise argparse.ArgumentTypeError(
            "Restricted MIPA/MIMIC inputs must be local filesystem paths; remote URLs are rejected."
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate locally generated MIPA phenotype predictions with subject-disjoint leakage audit, "
            "classification metrics, verbatim evidence checks, strict human-audit coverage, and explicit "
            "acceptance gates. This runner makes no model/API/network calls."
        )
    )
    parser.add_argument("--labels", required=True, type=_local_path, help="Local golden_labels.csv")
    parser.add_argument("--notes", required=True, type=_local_path, help="Local authorised golden_notes.csv")
    parser.add_argument(
        "--predictions",
        required=True,
        type=_local_path,
        help="Local CSV/JSONL predictions with note_id, phenotype, prediction, evidence, assertion, temporality",
    )
    parser.add_argument("--output-dir", required=True, type=_local_path)
    parser.add_argument(
        "--phenotypes",
        nargs="+",
        default=list(DEFAULT_PHENOTYPES),
        help="Phenotype columns to evaluate",
    )
    parser.add_argument(
        "--evaluation-scope",
        choices=["all", "train", "validation", "test"],
        default="all",
        help="Evaluate all rows or one deterministic subject-disjoint split",
    )
    parser.add_argument("--split-seed", default="20260723")
    parser.add_argument(
        "--evidence-audit",
        type=_local_path,
        help=(
            "Optional local human audit CSV with note_id, phenotype, supports_prediction, severe_context_error. "
            "Final PASS requires the configured coverage of positive predictions."
        ),
    )
    parser.add_argument("--macro-f1-threshold", type=float, default=0.85)
    parser.add_argument("--phenotype-f1-threshold", type=float, default=0.80)
    parser.add_argument("--min-phenotypes-passing", type=int, default=5)
    parser.add_argument("--common-recall-threshold", type=float, default=0.60)
    parser.add_argument("--common-positive-min", type=int, default=20)
    parser.add_argument("--evidence-verbatim-threshold", type=float, default=0.99)
    parser.add_argument("--evidence-support-threshold", type=float, default=0.90)
    parser.add_argument("--severe-context-error-threshold", type=float, default=0.05)
    parser.add_argument(
        "--min-human-audit-coverage",
        type=float,
        default=1.0,
        help="Required fraction of positive model predictions with human evidence audit; default 1.0 (100%).",
    )
    parser.add_argument(
        "--require-final-pass",
        action="store_true",
        help="Exit non-zero unless automated, human evidence quality, and audit coverage gates all pass.",
    )
    args = parser.parse_args()

    thresholds = AcceptanceThresholds(
        macro_f1=args.macro_f1_threshold,
        phenotype_f1=args.phenotype_f1_threshold,
        min_phenotypes_passing=args.min_phenotypes_passing,
        common_phenotype_recall=args.common_recall_threshold,
        common_positive_min=args.common_positive_min,
        evidence_verbatim_rate=args.evidence_verbatim_threshold,
        evidence_support_rate=args.evidence_support_threshold,
        severe_context_error_rate=args.severe_context_error_threshold,
    )
    audit_policy = EvidenceAuditCoveragePolicy(
        min_positive_prediction_coverage=args.min_human_audit_coverage
    )

    summary, metrics, errors, manifest = evaluate_mipa_predictions_strict(
        labels_path=args.labels,
        notes_path=args.notes,
        predictions_path=args.predictions,
        phenotypes=args.phenotypes,
        evaluation_scope=args.evaluation_scope,
        split_seed=args.split_seed,
        evidence_audit_path=args.evidence_audit,
        thresholds=thresholds,
        audit_policy=audit_policy,
    )
    write_evaluation_outputs(args.output_dir, summary, metrics, errors, manifest)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nOutputs: {Path(args.output_dir).resolve()}")

    status = str(summary["acceptance"]["final_status"])
    if args.require_final_pass and status != "PASS":
        raise SystemExit(f"Acceptance gate did not pass: {status}")


if __name__ == "__main__":
    main()
