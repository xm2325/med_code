from __future__ import annotations

from pathlib import Path

import pandas as pd

from .knowledge import prepare_terminology_knowledge


def prepare_icd10_terminology_from_mimic_dictionary(
    dictionary_path: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Convert MIMIC-IV d_icd_diagnoses into the MedCode terminology schema."""
    raw = pd.read_csv(dictionary_path, dtype=str, keep_default_na=False)
    required = {"icd_code", "icd_version", "long_title"}
    if not required.issubset(raw.columns):
        raise ValueError(f"ICD dictionary requires {sorted(required)}")
    raw = raw[raw["icd_version"].astype(str).str.strip() == "10"].copy()
    frame = pd.DataFrame({
        "system": "ICD-10",
        "code": raw["icd_code"].astype(str).str.strip(),
        "term": raw["long_title"].astype(str),
        "synonyms": "",
        "definition": "",
        "hierarchy": "",
        "knowledge_source": "MIMIC-IV d_icd_diagnoses",
    })
    frame = frame[frame["code"].str.len() > 0].drop_duplicates("code").reset_index(drop=True)
    prepared = prepare_terminology_knowledge(frame, coding_system="ICD-10")
    if output_csv is not None:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(output, index=False)
    return prepared
