"""Three-way clinician-gold × text-model × structured-record discordance analysis."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .mipa_phenotyping import DEFAULT_PHENOTYPES, _binary, _read_predictions, _require_columns


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _load_upstream_status(path: str | Path | None) -> str | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(payload.get("acceptance", {}).get("final_status", "UNKNOWN"))


def evaluate_three_way_discordance(
    *,
    labels_path: str | Path,
    text_predictions_path: str | Path,
    structured_status_path: str | Path,
    phenotypes: Sequence[str] = DEFAULT_PHENOTYPES,
    upstream_summary_path: str | Path | None = None,
    structured_scope_validated: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Compare gold phenotype labels (G), text predictions (T), and structured status (C).

    `structured_scope_validated` means the structured-code definition has been reviewed to
    match the phenotype target. Without it, discordance is reported but under-recording
    interpretation remains exploratory.
    """
    labels = _read_csv(labels_path)
    text_predictions = _read_predictions(text_predictions_path)
    structured = _read_csv(structured_status_path)
    _require_columns(labels, ["note_id", "subject_id", *phenotypes], "Gold labels")
    _require_columns(text_predictions, ["note_id", "phenotype", "prediction"], "Text predictions")
    _require_columns(structured, ["note_id", "phenotype", "structured_positive"], "Structured status")

    label_by_note = {str(row["note_id"]): row for row in labels}
    if len(label_by_note) != len(labels):
        raise ValueError("Gold labels contain duplicate note_id values")

    text_key_counts = Counter((str(row["note_id"]), str(row["phenotype"])) for row in text_predictions)
    structured_key_counts = Counter((str(row["note_id"]), str(row["phenotype"])) for row in structured)
    if any(count > 1 for count in text_key_counts.values()):
        raise ValueError("Text predictions contain duplicate note_id/phenotype keys")
    if any(count > 1 for count in structured_key_counts.values()):
        raise ValueError("Structured status contains duplicate note_id/phenotype keys")

    text_by_key = {(str(row["note_id"]), str(row["phenotype"])): row for row in text_predictions}
    structured_by_key = {(str(row["note_id"]), str(row["phenotype"])): row for row in structured}

    per_phenotype: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    expected = len(labels) * len(phenotypes)
    observed_text = 0
    observed_structured = 0

    totals = Counter()
    for phenotype in phenotypes:
        cells = Counter()
        for note_id, label_row in label_by_note.items():
            key = (note_id, phenotype)
            text_row = text_by_key.get(key)
            structured_row = structured_by_key.get(key)
            if text_row is None or structured_row is None:
                continue
            observed_text += 1
            observed_structured += 1
            g = _binary(label_row[phenotype])
            t = _binary(text_row["prediction"])
            c = _binary(structured_row["structured_positive"])
            cells[(g, t, c)] += 1
            totals[(g, t, c)] += 1

        gold_positive = sum(count for (g, _, _), count in cells.items() if g == 1)
        structured_true_positive = sum(count for (g, _, c), count in cells.items() if g == 1 and c == 1)
        text_true_positive = sum(count for (g, t, _), count in cells.items() if g == 1 and t == 1)
        combined_true_positive = sum(count for (g, t, c), count in cells.items() if g == 1 and (t == 1 or c == 1))
        code_missed_gold = cells[(1, 0, 0)] + cells[(1, 1, 0)]
        recovered_by_text = cells[(1, 1, 0)]
        text_positive_code_absent = cells[(0, 1, 0)] + cells[(1, 1, 0)]

        per_phenotype.append(
            {
                "phenotype": phenotype,
                "n_evaluable": sum(cells.values()),
                "gold_positive": gold_positive,
                "structured_sensitivity": _ratio(structured_true_positive, gold_positive),
                "text_sensitivity": _ratio(text_true_positive, gold_positive),
                "combined_sensitivity": _ratio(combined_true_positive, gold_positive),
                "gold_positive_code_absent": code_missed_gold,
                "gold_positive_text_positive_code_absent": recovered_by_text,
                "recoverable_code_missed_fraction": _ratio(recovered_by_text, code_missed_gold),
                "gold_ppv_among_text_positive_code_absent": _ratio(recovered_by_text, text_positive_code_absent),
            }
        )
        for g in (0, 1):
            for t in (0, 1):
                for c in (0, 1):
                    cell_rows.append(
                        {
                            "phenotype": phenotype,
                            "gold": g,
                            "text": t,
                            "structured": c,
                            "n": cells[(g, t, c)],
                        }
                    )

    text_coverage = _ratio(observed_text, expected)
    structured_coverage = _ratio(observed_structured, expected)
    upstream_status = _load_upstream_status(upstream_summary_path)
    upstream_pass = upstream_status == "PASS"
    complete_coverage = text_coverage == 1.0 and structured_coverage == 1.0

    confirmatory_eligible = upstream_pass and structured_scope_validated and complete_coverage
    if confirmatory_eligible:
        interpretation_status = "CONFIRMATORY_ELIGIBLE"
    else:
        interpretation_status = "EXPLORATORY_DISCORDANCE_ONLY"

    overall_gold_positive = sum(count for (g, _, _), count in totals.items() if g == 1)
    overall_structured_tp = sum(count for (g, _, c), count in totals.items() if g == 1 and c == 1)
    overall_text_tp = sum(count for (g, t, _), count in totals.items() if g == 1 and t == 1)
    overall_combined_tp = sum(count for (g, t, c), count in totals.items() if g == 1 and (t == 1 or c == 1))
    overall_code_missed = totals[(1, 0, 0)] + totals[(1, 1, 0)]
    overall_recovered = totals[(1, 1, 0)]
    overall_text_pos_code_absent = totals[(0, 1, 0)] + totals[(1, 1, 0)]

    summary: dict[str, object] = {
        "schema_version": "gold-text-structured-discordance-v0.3.0",
        "definitions": {
            "G": "clinician/expert gold phenotype label",
            "T": "text-model phenotype prediction",
            "C": "structured-record phenotype indicator",
            "candidate_underrecording_cell": "G=1,T=1,C=0",
        },
        "coverage": {
            "expected_note_phenotype_pairs": expected,
            "text_prediction_coverage": text_coverage,
            "structured_status_coverage": structured_coverage,
            "complete": complete_coverage,
        },
        "upstream_gate": {
            "stage1_2_summary_supplied": upstream_summary_path is not None,
            "stage1_2_final_status": upstream_status,
            "stage1_2_pass": upstream_pass,
        },
        "structured_definition": {
            "scope_validated": structured_scope_validated,
        },
        "overall": {
            "gold_positive": overall_gold_positive,
            "structured_sensitivity": _ratio(overall_structured_tp, overall_gold_positive),
            "text_sensitivity": _ratio(overall_text_tp, overall_gold_positive),
            "combined_sensitivity": _ratio(overall_combined_tp, overall_gold_positive),
            "gold_positive_code_absent": overall_code_missed,
            "gold_positive_text_positive_code_absent": overall_recovered,
            "recoverable_code_missed_fraction": _ratio(overall_recovered, overall_code_missed),
            "gold_ppv_among_text_positive_code_absent": _ratio(overall_recovered, overall_text_pos_code_absent),
        },
        "interpretation": {
            "status": interpretation_status,
            "confirmatory_underrecording_language_allowed": confirmatory_eligible,
            "reason": (
                "Stage 1/2 final PASS, validated structured phenotype scope, and complete pair coverage are all required "
                "before G=1,T=1,C=0 is treated as confirmatory recoverable under-recording."
            ),
            "negative_result_is_valid": True,
            "no_minimum_underrecording_effect_required_for_success": True,
        },
    }
    return summary, per_phenotype, cell_rows


def write_discordance_outputs(
    output_dir: str | Path,
    summary: Mapping[str, object],
    per_phenotype: Sequence[Mapping[str, object]],
    cells: Sequence[Mapping[str, object]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metric_fields = [
        "phenotype", "n_evaluable", "gold_positive", "structured_sensitivity", "text_sensitivity",
        "combined_sensitivity", "gold_positive_code_absent", "gold_positive_text_positive_code_absent",
        "recoverable_code_missed_fraction", "gold_ppv_among_text_positive_code_absent",
    ]
    with (output_dir / "phenotype_discordance_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(per_phenotype)

    with (output_dir / "discordance_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phenotype", "gold", "text", "structured", "n"])
        writer.writeheader()
        writer.writerows(cells)
