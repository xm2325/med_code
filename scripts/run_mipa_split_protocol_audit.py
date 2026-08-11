#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_LABELS_URL = (
    "https://raw.githubusercontent.com/open-health-data-lab/MIPA/main/"
    "data/filters/golden_labels.csv"
)

PHENOTYPES = [
    "hypertension",
    "depression",
    "diabetes_type_2",
    "hfpef",
    "vte_past",
    "obesity",
    "rheumatoid_arthritis",
]


def _load_labels(path: str | None, url: str) -> pd.DataFrame:
    df = pd.read_csv(path or url)
    required = {"note_id", "subject_id", "hadm_id", *PHENOTYPES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"MIPA labels missing columns: {missing}")
    if df["note_id"].duplicated().any():
        raise ValueError("Duplicate note_id values found")
    return df


def official_mipa_admission_split(df: pd.DataFrame, phenotype: str, val_frac: float = 0.4):
    """Mirror public MIPA phenotype_cohort_test splitting logic.

    The official implementation samples positive and negative admission rows separately with
    pandas.DataFrame.sample(frac=0.4, random_state=42), then assigns the remaining admissions
    to test. It checks hadm_id overlap, not subject-level overlap.
    """
    cohort = df.loc[df[phenotype].notna(), ["note_id", "subject_id", "hadm_id", phenotype]].copy()
    cohort[phenotype] = pd.to_numeric(cohort[phenotype], errors="coerce").fillna(0).astype(int)
    cohort = cohort.drop_duplicates(subset=["hadm_id"])
    positives = cohort[cohort[phenotype] == 1]
    negatives = cohort[cohort[phenotype] == 0]
    pos_val = positives.sample(frac=val_frac, random_state=42) if len(positives) else positives
    neg_val = negatives.sample(frac=val_frac, random_state=42) if len(negatives) else negatives
    val = pd.concat([pos_val, neg_val]).sample(frac=1, random_state=42).reset_index(drop=True)
    test = cohort.drop(index=pos_val.index.union(neg_val.index)).reset_index(drop=True)
    return val, test


def subject_overlap_audit(val: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    val_subjects = set(val["subject_id"].astype(str))
    test_subjects = set(test["subject_id"].astype(str))
    overlap = val_subjects & test_subjects
    val_overlap_rows = int(val["subject_id"].astype(str).isin(overlap).sum())
    test_overlap_rows = int(test["subject_id"].astype(str).isin(overlap).sum())
    return {
        "n_validation_rows": int(len(val)),
        "n_test_rows": int(len(test)),
        "n_validation_subjects": len(val_subjects),
        "n_test_subjects": len(test_subjects),
        "n_overlapping_subjects": len(overlap),
        "fraction_validation_subjects_overlapping_test": len(overlap) / len(val_subjects) if val_subjects else None,
        "fraction_test_subjects_overlapping_validation": len(overlap) / len(test_subjects) if test_subjects else None,
        "validation_rows_from_overlapping_subjects": val_overlap_rows,
        "test_rows_from_overlapping_subjects": test_overlap_rows,
        "hadm_overlap": int(len(set(val["hadm_id"]) & set(test["hadm_id"]))),
        "subject_disjoint": len(overlap) == 0,
    }


def deterministic_subject_split(df: pd.DataFrame, seed: str = "20260723") -> pd.DataFrame:
    from cohortcoder.mipa_phenotyping import assign_subject_split

    out = df[["note_id", "subject_id", "hadm_id"]].copy()
    out["split"] = out["subject_id"].map(lambda value: assign_subject_split(value, seed=seed))
    return out


def subject_disjoint_audit(manifest: pd.DataFrame) -> dict[str, object]:
    subject_sets = {
        split: set(group["subject_id"].astype(str))
        for split, group in manifest.groupby("split")
    }
    overlaps: dict[str, int] = {}
    names = sorted(subject_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlaps[f"{left}__{right}"] = len(subject_sets[left] & subject_sets[right])
    return {
        "rows_per_split": manifest["split"].value_counts().sort_index().to_dict(),
        "subjects_per_split": {
            split: len(subjects) for split, subjects in sorted(subject_sets.items())
        },
        "subject_overlap_counts": overlaps,
        "subject_disjoint": all(value == 0 for value in overlaps.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MIPA benchmark-comparison and RA confirmatory split protocols")
    parser.add_argument("--labels-path")
    parser.add_argument("--labels-url", default=DEFAULT_LABELS_URL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", default="20260723")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_labels(args.labels_path, args.labels_url)

    official_rows = []
    official_detail = {}
    for phenotype in PHENOTYPES:
        val, test = official_mipa_admission_split(df, phenotype)
        audit = subject_overlap_audit(val, test)
        official_detail[phenotype] = audit
        official_rows.append({"phenotype": phenotype, **audit})

    ra = df[pd.to_numeric(df["rheumatoid_arthritis"], errors="coerce").fillna(0).astype(int) == 1].copy()
    ra_manifest = deterministic_subject_split(ra, seed=args.split_seed)
    ra_subject_audit = subject_disjoint_audit(ra_manifest)

    any_official_subject_overlap = any(not row["subject_disjoint"] for row in official_rows)
    summary = {
        "schema_version": "mipa-split-protocol-audit-v0.3.1",
        "source": {
            "labels": args.labels_path or args.labels_url,
            "official_reference": "open-health-data-lab/MIPA scripts/preprocessing/cohort.py phenotype_cohort_test",
        },
        "benchmark_comparison_protocol": {
            "name": "official_mipa_admission_level_40_60",
            "purpose": "Reproduce/compare with MIPA admission-level validation/test protocol where applicable.",
            "implementation": "Per phenotype, positive and negative hadm_id rows are sampled separately into 40% validation using pandas sample(random_state=42); remaining rows are test.",
            "subject_grouped": False,
            "any_subject_overlap_between_validation_and_test": any_official_subject_overlap,
            "per_phenotype": official_detail,
            "interpretation": "Use only for protocol-matched benchmark comparison. Do not treat admission-disjointness as patient-disjointness.",
        },
        "ra_confirmatory_protocol": {
            "name": "deterministic_subject_disjoint",
            "purpose": "Primary RA scientific confirmation and leakage-resistant error analysis.",
            "n_ra_positive_notes": int(len(ra)),
            "n_ra_unique_subjects": int(ra["subject_id"].nunique()),
            "split_seed": args.split_seed,
            "audit": ra_subject_audit,
            "interpretation": "All notes from one subject remain in one split; confirmatory uncertainty should resample subject_id clusters.",
        },
        "reporting_rule": {
            "do_not_mix_protocol_metrics": True,
            "published_external_macro_f1_directly_comparable_to_our_subject_disjoint_result": False,
            "reason": "A performance number is directly comparable only when model, phenotype definition, evaluation cohort, and split/evaluation protocol are matched.",
        },
    }

    pd.DataFrame(official_rows).to_csv(output_dir / "official_split_subject_overlap.csv", index=False)
    ra_manifest.to_csv(output_dir / "ra_subject_disjoint_manifest.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
