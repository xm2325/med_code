from __future__ import annotations

from pathlib import Path

import pandas as pd


OPTIONAL_KNOWLEDGE_COLUMNS = ["system", "synonyms", "definition", "hierarchy", "knowledge_source"]


def prepare_terminology_knowledge(
    terminology: pd.DataFrame,
    *,
    coding_system: str | None = None,
) -> pd.DataFrame:
    """Prepare a terminology table without destroying human-readable term labels.

    One run intentionally targets one coding system. ICD and MedDRA can use the same
    pipeline, but mixing both systems in one candidate dictionary makes identical code
    strings and different hierarchy semantics difficult to audit.
    """
    df = terminology.copy().fillna("")
    if not {"code", "term"}.issubset(df.columns):
        raise ValueError("terminology requires code and term columns")
    for column in OPTIONAL_KNOWLEDGE_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df["code"] = df["code"].astype(str)
    df["term"] = df["term"].astype(str)
    df["system"] = df["system"].astype(str)

    if coding_system:
        if not (df["system"].str.strip().str.len() > 0).any():
            df["system"] = str(coding_system)
        else:
            df = df[df["system"].str.lower() == str(coding_system).lower()].copy()
            if df.empty:
                raise ValueError(f"No terminology rows found for coding system: {coding_system}")

    systems = sorted({value.strip() for value in df["system"].astype(str) if value.strip()})
    if len(systems) > 1:
        raise ValueError(
            "A benchmark must use one coding system at a time. Filter the terminology "
            f"to one system; found: {systems}"
        )

    # Search text can use synonyms and definitions, while `term` stays clean for display.
    search_parts = [df["term"].astype(str)]
    search_parts.append(df["synonyms"].astype(str).str.replace("|", " ", regex=False))
    search_parts.append(df["definition"].astype(str))
    df["search_text"] = search_parts[0]
    for part in search_parts[1:]:
        df["search_text"] = df["search_text"].str.cat(part, sep=" ")
    df["search_text"] = df["search_text"].str.replace(r"\s+", " ", regex=True).str.strip()

    columns = ["code", "term", *OPTIONAL_KNOWLEDGE_COLUMNS, "search_text"]
    return df[columns].drop_duplicates("code").reset_index(drop=True)


def load_terminology_knowledge(
    path: str | Path,
    *,
    coding_system: str | None = None,
) -> pd.DataFrame:
    return prepare_terminology_knowledge(pd.read_csv(path), coding_system=coding_system)
