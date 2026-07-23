#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

DEFAULT_LABELS_URL = (
    "https://raw.githubusercontent.com/open-health-data-lab/MIPA/main/"
    "data/filters/golden_labels.csv"
)

PHENOTYPES = [
    "alcohol_abuse",
    "c_diff_complication",
    "c_diff_past",
    "dementia",
    "depression",
    "diabetes_type_1",
    "diabetes_type_2",
    "hfpef",
    "hfref",
    "hypertension",
    "sle",
    "metastatic_cancer",
    "none",
    "obesity",
    "rheumatoid_arthritis",
    "vte_complication",
    "vte_past",
]


def _download_text(url: str) -> str:
    request = Request(
        url,
        headers={"User-Agent": "med-code-mipa-public-pilot/0.3"},
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed/explicit research URL
        return response.read().decode("utf-8-sig")


def _load_rows(labels_path: str | None, labels_url: str) -> list[dict[str, str]]:
    if labels_path:
        text = Path(labels_path).read_text(encoding="utf-8-sig")
    else:
        text = _download_text(labels_url)
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise RuntimeError("MIPA golden-label file is empty")
    missing = [column for column in ["note_id", "subject_id", *PHENOTYPES] if column not in rows[0]]
    if missing:
        raise RuntimeError(f"MIPA golden-label schema missing columns: {missing}")
    return rows


def _is_positive(value: str | None) -> bool:
    if value is None:
        return False
    value = str(value).strip()
    if not value:
        return False
    try:
        return float(value) == 1.0
    except ValueError:
        return value.lower() in {"true", "yes", "positive"}


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _repetition_summary(values: list[str]) -> dict[str, int]:
    counts = Counter(values)
    return {
        "n_rows": len(values),
        "n_unique": len(counts),
        "n_ids_with_multiple_rows": sum(count > 1 for count in counts.values()),
        "max_rows_per_id": max(counts.values()) if counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the public, labels-only MIPA feasibility audit for the RA comorbidity pilot. "
            "This does not require or reconstruct restricted MIMIC-IV discharge summaries."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels-path", help="Optional local MIPA golden_labels.csv")
    parser.add_argument("--labels-url", default=DEFAULT_LABELS_URL)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.labels_path, args.labels_url)
    note_ids = [str(row["note_id"]) for row in rows]
    subject_ids = [str(row["subject_id"]) for row in rows]

    duplicate_note_ids = [note_id for note_id, count in Counter(note_ids).items() if count > 1]
    if duplicate_note_ids:
        raise RuntimeError(f"Duplicate note_id values found: {duplicate_note_ids[:5]}")

    phenotype_counts = []
    for phenotype in PHENOTYPES:
        count = sum(_is_positive(row.get(phenotype)) for row in rows)
        phenotype_counts.append(
            {
                "phenotype": phenotype,
                "positive_notes": count,
                "positive_fraction_of_benchmark": count / len(rows),
            }
        )

    ra_rows = [row for row in rows if _is_positive(row.get("rheumatoid_arthritis"))]
    ra_subject_ids = [str(row["subject_id"]) for row in ra_rows]

    ra_comorbidity_counts = []
    for phenotype in PHENOTYPES:
        if phenotype in {"rheumatoid_arthritis", "none"}:
            continue
        count = sum(_is_positive(row.get(phenotype)) for row in ra_rows)
        ra_comorbidity_counts.append(
            {
                "phenotype": phenotype,
                "positive_among_ra_notes": count,
                "fraction_among_ra_benchmark_notes": (count / len(ra_rows)) if ra_rows else None,
            }
        )
    ra_comorbidity_counts.sort(
        key=lambda item: (-int(item["positive_among_ra_notes"]), str(item["phenotype"]))
    )

    ra_cooccurrence_burden = Counter()
    for row in ra_rows:
        n_other = sum(
            _is_positive(row.get(phenotype))
            for phenotype in PHENOTYPES
            if phenotype not in {"rheumatoid_arthritis", "none"}
        )
        ra_cooccurrence_burden[n_other] += 1

    ra_dataset_sources = Counter(str(row.get("dataset_name", "")) for row in ra_rows)

    summary = {
        "study": "MIPA public-label feasibility audit for RA comorbidity phenotyping",
        "source": {
            "labels_url": args.labels_url if not args.labels_path else None,
            "labels_path": args.labels_path,
            "upstream_repository": "open-health-data-lab/MIPA",
            "upstream_file": "data/filters/golden_labels.csv",
        },
        "benchmark": {
            "n_discharge_summary_labels": len(rows),
            "n_phenotype_columns": len(PHENOTYPES),
            "phenotypes": PHENOTYPES,
            "note_id_audit": _repetition_summary(note_ids),
            "subject_id_audit": _repetition_summary(subject_ids),
        },
        "ra_subset": {
            "n_ra_positive_notes": len(ra_rows),
            "subject_id_audit": _repetition_summary(ra_subject_ids),
            "candidate_source_counts": dict(sorted(ra_dataset_sources.items())),
            "cooccurring_phenotype_count_distribution": {
                str(key): value for key, value in sorted(ra_cooccurrence_burden.items())
            },
        },
        "interpretation": {
            "public_labels_reproducible": True,
            "discharge_summary_text_publicly_available": False,
            "deepseek_mipa_note_evaluation_executed": False,
            "why_not": (
                "MIPA expert labels are public, but the underlying MIMIC-IV discharge summaries require "
                "credentialed PhysioNet access. A real DeepSeek note-level experiment must use an authorised "
                "local copy of golden_notes.csv and must not infer or reconstruct notes from labels."
            ),
            "ra_comorbidity_fractions_are_prevalence_estimates": False,
            "sampling_warning": (
                "MIPA is a phenotype benchmark assembled from phenotype-targeted candidate notes; RA-subset "
                "co-occurrence fractions describe this benchmark sample only and must not be reported as RA prevalence."
            ),
        },
    }

    _write_csv(
        output_dir / "phenotype_counts.csv",
        ["phenotype", "positive_notes", "positive_fraction_of_benchmark"],
        phenotype_counts,
    )
    _write_csv(
        output_dir / "ra_comorbidity_counts.csv",
        ["phenotype", "positive_among_ra_notes", "fraction_among_ra_benchmark_notes"],
        ra_comorbidity_counts,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nTop RA-benchmark co-occurring phenotypes:")
    for item in ra_comorbidity_counts[:10]:
        print(
            f"  {item['phenotype']}: {item['positive_among_ra_notes']} "
            f"({item['fraction_among_ra_benchmark_notes']:.1%})"
        )


if __name__ == "__main__":
    main()
