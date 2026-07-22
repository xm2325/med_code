from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_coding_records(path: str | Path) -> pd.DataFrame:
    """Load normalized coding records while preserving document-level semantics.

    A missing `mention` means no pre-extracted mention is available, so it is stored as
    an empty string and downstream code falls back to the full text. It is deliberately
    not replaced with the entire document, because doing so would make the whole note a
    trivial rationale span.
    """
    df = pd.read_csv(path).fillna("")
    if not {"record_id", "text"}.issubset(df.columns):
        raise ValueError("records require record_id and text columns")
    if "mention" not in df.columns:
        df["mention"] = ""
    if "gold_code" not in df.columns:
        df["gold_code"] = ""
    if "gold_term" not in df.columns:
        df["gold_term"] = ""
    for column in ["record_id", "text", "mention", "gold_code", "gold_term"]:
        df[column] = df[column].astype(str)
    return df
