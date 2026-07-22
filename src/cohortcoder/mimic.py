from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
from typing import Any

import pandas as pd


def _read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def assign_subject_splits(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    train: float = 0.70,
    val: float = 0.15,
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """Assign deterministic patient-level splits.

    All admissions/notes for one subject stay in the same split. This is stricter than
    document-level splitting and avoids repeated-patient leakage in MIMIC experiments.
    """
    if subject_col not in df.columns:
        raise ValueError(f"Missing split column: {subject_col}")
    if not (0 < train < 1 and 0 <= val < 1 and train + val < 1):
        raise ValueError("train/val fractions must satisfy 0 < train and train + val < 1")
    out = df.copy()
    mapping: dict[str, str] = {}
    for subject in out[subject_col].astype(str).unique():
        u = int.from_bytes(sha256(f"{seed}:{subject}".encode()).digest()[:8], "big") / 2**64
        mapping[subject] = "train" if u < train else ("val" if u < train + val else "test")
    out["split"] = out[subject_col].astype(str).map(mapping)
    return out


def assert_subject_disjoint(df: pd.DataFrame, *, subject_col: str = "subject_id") -> None:
    if "split" not in df.columns:
        raise ValueError("split column missing")
    if subject_col not in df.columns:
        raise ValueError(f"{subject_col} column missing")
    memberships = df.groupby(subject_col)["split"].nunique()
    if (memberships > 1).any():
        raise ValueError("patient leakage: subject occurs in multiple splits")


def prepare_mimic_iv_icd10(
    discharge_path: str | Path,
    diagnoses_path: str | Path,
    dictionary_path: str | Path,
    output_csv: str | Path,
    *,
    min_code_frequency: int = 1,
    max_notes: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a patient-split, multi-label ICD-10 discharge-summary benchmark table.

    Expected source tables are the MIMIC-IV-Note ``discharge`` table and MIMIC-IV
    ``diagnoses_icd`` / ``d_icd_diagnoses`` tables. Only ``icd_version == 10`` rows
    are retained. One row is emitted per hospitalization, with JSON arrays of diagnosis
    codes and long titles.

    No source text is safe to publish merely because this function created a derived
    file. MIMIC data and derivatives remain governed by the source data agreement.
    """
    discharge = _read_table(discharge_path)
    diagnoses = _read_table(diagnoses_path)
    dictionary = _read_table(dictionary_path)

    required_discharge = {"subject_id", "hadm_id", "text"}
    required_diag = {"subject_id", "hadm_id", "icd_code", "icd_version"}
    required_dict = {"icd_code", "icd_version", "long_title"}
    if not required_discharge.issubset(discharge.columns):
        raise ValueError(f"discharge table requires {sorted(required_discharge)}")
    if not required_diag.issubset(diagnoses.columns):
        raise ValueError(f"diagnoses table requires {sorted(required_diag)}")
    if not required_dict.issubset(dictionary.columns):
        raise ValueError(f"diagnosis dictionary requires {sorted(required_dict)}")

    discharge = discharge[discharge["hadm_id"].astype(str).str.len() > 0].copy()
    # A hospitalization should have one canonical discharge summary for this benchmark.
    # If multiple rows exist, keep the latest deterministically using available times.
    sort_cols = [col for col in ["charttime", "storetime", "note_id"] if col in discharge.columns]
    if sort_cols:
        discharge = discharge.sort_values(sort_cols)
    discharge = discharge.drop_duplicates("hadm_id", keep="last")

    diagnoses = diagnoses[diagnoses["icd_version"].astype(str).str.strip() == "10"].copy()
    dictionary = dictionary[dictionary["icd_version"].astype(str).str.strip() == "10"].copy()
    dictionary = dictionary.drop_duplicates(["icd_code", "icd_version"])
    diagnoses = diagnoses.merge(
        dictionary[["icd_code", "icd_version", "long_title"]],
        on=["icd_code", "icd_version"],
        how="left",
        validate="many_to_one",
    )

    diagnoses["icd_code"] = diagnoses["icd_code"].astype(str).str.strip()
    diagnoses = diagnoses[diagnoses["icd_code"].str.len() > 0]
    if min_code_frequency > 1:
        counts = diagnoses["icd_code"].value_counts()
        keep = set(counts[counts >= int(min_code_frequency)].index.astype(str))
        diagnoses = diagnoses[diagnoses["icd_code"].isin(keep)]

    grouped_rows: list[dict[str, Any]] = []
    for hadm_id, group in diagnoses.groupby("hadm_id", sort=False):
        codes: list[str] = []
        titles: list[str] = []
        seen: set[str] = set()
        for _, row in group.iterrows():
            code = str(row["icd_code"])
            if code in seen:
                continue
            seen.add(code)
            codes.append(code)
            titles.append(str(row.get("long_title", "")))
        if codes:
            grouped_rows.append({
                "hadm_id": str(hadm_id),
                "gold_codes_json": json.dumps(codes),
                "gold_terms_json": json.dumps(titles, ensure_ascii=False),
                "n_gold_codes": len(codes),
            })
    gold = pd.DataFrame(grouped_rows)
    if gold.empty:
        raise ValueError("No ICD-10 diagnosis labels remained after filtering")

    merged = discharge.merge(gold, on="hadm_id", how="inner", validate="one_to_one")
    if max_notes is not None:
        merged = merged.head(int(max_notes)).copy()
    merged["record_id"] = merged["hadm_id"].astype(str)
    merged["source_dataset"] = "MIMIC-IV-Note+MIMIC-IV"
    keep_cols = [
        "record_id", "subject_id", "hadm_id",
        *(["note_id"] if "note_id" in merged.columns else []),
        "text", "gold_codes_json", "gold_terms_json", "n_gold_codes", "source_dataset",
    ]
    out = assign_subject_splits(merged[keep_cols], seed=seed)
    assert_subject_disjoint(out)

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    stats = {
        "rows": int(len(out)),
        "subjects": int(out["subject_id"].nunique()),
        "unique_icd10_codes": int(len({code for value in out["gold_codes_json"] for code in json.loads(value)})),
        "mean_codes_per_note": float(out["n_gold_codes"].astype(float).mean()),
        "split_counts": out["split"].value_counts().to_dict(),
        "split_unit": "subject_id",
        "icd_version": 10,
        "min_code_frequency": int(min_code_frequency),
    }
    output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out
