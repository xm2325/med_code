from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd


OFFICIAL_MEDNORM = {
    "name": "MedNorm: A Corpus and Embeddings for Cross-terminology Medical Concept Normalisation",
    "doi": "10.17632/b9x7xxb9sz.1",
    "official_url": "https://data.mendeley.com/datasets/b9x7xxb9sz/1",
    "licence": "CC BY-NC 3.0",
    "authors": "Maksim Belousov; William G. Dixon; Goran Nenadic",
    "n_reported_rows": 27979,
}


def fetch_hf_mirror_rows(
    *,
    dataset: str = "awacke1/MedNorm2SnomedCT2UMLS",
    config: str = "default",
    split: str = "train",
    max_rows: int | None = None,
    page_size: int = 100,
) -> pd.DataFrame:
    """Fetch a convenience mirror through the Hugging Face dataset-viewer rows API.

    The mirror is not treated as the licensing authority. `OFFICIAL_MEDNORM` remains the
    source/licence record and must be cited in reports. This helper exists to make a public,
    reproducible evaluation path easier when the official archive UI is inconvenient.
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        length = min(page_size, (max_rows - len(rows)) if max_rows is not None else page_size)
        if length <= 0:
            break
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=" + quote(dataset, safe="")
            + "&config=" + quote(config, safe="")
            + "&split=" + quote(split, safe="")
            + f"&offset={offset}&length={length}"
        )
        with urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        page = [dict(item.get("row", {})) for item in payload.get("rows", [])]
        rows.extend(page)
        if not page or len(rows) >= int(payload.get("num_rows_total", len(rows))):
            break
        offset += len(page)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows[:max_rows] if max_rows is not None else rows)


def prepare_mednorm_single_meddra(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"original_dataset", "instance_id", "phrase", "single_mapped_meddra_codes"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Missing MedNorm columns: " + ",".join(sorted(missing)))
    out = frame.copy()
    out["phrase"] = out["phrase"].astype(str).str.strip()
    out["gold_code"] = out["single_mapped_meddra_codes"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    out = out[(out["phrase"] != "") & (out["gold_code"] != "")].copy()
    out["record_id"] = out["instance_id"].astype(str)
    out["text"] = out["phrase"]
    out["mention"] = out["phrase"]
    out["gold_term"] = ""
    out["source_dataset"] = out["original_dataset"].astype(str)
    return out[["record_id", "text", "mention", "gold_code", "gold_term", "source_dataset"]].drop_duplicates()


def assign_cross_dataset_split(records: pd.DataFrame, *, test_source: str = "CADEC", val_fraction: float = 0.15) -> pd.DataFrame:
    data = records.copy()
    data["split"] = "train"
    data.loc[data["source_dataset"].astype(str).str.lower() == str(test_source).lower(), "split"] = "test"
    train_mask = data["split"] == "train"
    def is_val(record_id: str) -> bool:
        value = int(hashlib.sha256(str(record_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return value < float(val_fraction)
    data.loc[train_mask & data["record_id"].astype(str).map(is_val), "split"] = "val"
    if not {"train", "val", "test"}.issubset(set(data["split"])):
        raise ValueError("Cross-dataset split requires non-empty train/val/test")
    return data


def build_train_derived_terminology(train: pd.DataFrame, *, max_aliases_per_code: int = 50) -> pd.DataFrame:
    """Closed-code diagnostic terminology from TRAIN only; never use TEST phrases here."""
    rows = []
    for code, group in train.groupby("gold_code", sort=True):
        phrases = list(dict.fromkeys(group["mention"].astype(str).tolist()))[:max_aliases_per_code]
        if not phrases:
            continue
        rows.append({
            "system": "MedDRA",
            "code": str(code),
            "term": phrases[0],
            "synonyms": " | ".join(phrases[1:]),
            "definition": "",
            "hierarchy": "",
            "knowledge_source": "MedNorm TRAIN-derived aliases; closed-code diagnostic only",
        })
    return pd.DataFrame(rows)


def mednorm_data_card() -> dict[str, Any]:
    return {
        **OFFICIAL_MEDNORM,
        "task_use": "medical concept normalisation benchmark",
        "licence_authority": "official Mendeley Data record",
        "mirror_note": "A third-party mirror may be used only as a transport convenience; it does not override the official licence.",
    }
