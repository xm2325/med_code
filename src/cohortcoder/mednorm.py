from __future__ import annotations

from io import BytesIO
import hashlib
import json
from typing import Any, Iterable, Mapping
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


def _select_tabular_repo_file(siblings: Iterable[Mapping[str, Any]]) -> str | None:
    """Choose a likely source CSV/TSV from a Hugging Face repository listing."""
    candidates: list[str] = []
    for item in siblings:
        name = str(item.get("rfilename", "") or "")
        lower = name.lower()
        if not name or lower.startswith(".") or lower.endswith("readme.md"):
            continue
        if lower.endswith((".csv", ".tsv", ".txt")):
            candidates.append(name)
    if not candidates:
        return None
    # Prefer filenames that look like the dataset itself, then CSV, then shortest path.
    return sorted(
        candidates,
        key=lambda name: (
            0 if "mednorm" in name.lower() else 1,
            0 if name.lower().endswith(".csv") else 1,
            name.count("/"),
            len(name),
            name,
        ),
    )[0]


def fetch_hf_mirror_file(
    *,
    dataset: str = "awacke1/MedNorm2SnomedCT2UMLS",
    revision: str = "main",
    timeout_seconds: int = 60,
) -> pd.DataFrame:
    """Download the small public mirror as one tabular file instead of hundreds of row-API calls.

    The Hugging Face copy is only a transport convenience. The official Mendeley record in
    ``OFFICIAL_MEDNORM`` remains the source/licence authority.
    """
    api_url = "https://huggingface.co/api/datasets/" + quote(dataset, safe="/")
    with urlopen(api_url, timeout=timeout_seconds) as response:
        metadata = json.loads(response.read().decode("utf-8"))
    filename = _select_tabular_repo_file(metadata.get("siblings", []))
    if not filename:
        raise FileNotFoundError("No CSV/TSV source file found in the Hugging Face mirror")
    download_url = (
        "https://huggingface.co/datasets/"
        + quote(dataset, safe="/")
        + "/resolve/"
        + quote(revision, safe="")
        + "/"
        + quote(filename, safe="/")
        + "?download=true"
    )
    with urlopen(download_url, timeout=timeout_seconds) as response:
        raw = response.read()
    suffix = filename.lower().rsplit(".", 1)[-1]
    sep = "\t" if suffix in {"tsv", "txt"} else ","
    return pd.read_csv(BytesIO(raw), sep=sep, dtype=str, keep_default_na=False)


def fetch_hf_mirror_rows(
    *,
    dataset: str = "awacke1/MedNorm2SnomedCT2UMLS",
    config: str = "default",
    split: str = "train",
    max_rows: int | None = None,
    page_size: int = 100,
) -> pd.DataFrame:
    """Fallback dataset-viewer row API.

    Prefer ``fetch_hf_mirror_file`` for normal runs. This paginated fallback remains useful if
    the mirror repository layout changes.
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


def fetch_hf_mirror_dataframe(*, max_rows: int | None = None) -> pd.DataFrame:
    """Fast mirror loader with a conservative rows-API fallback."""
    try:
        frame = fetch_hf_mirror_file()
        return frame.head(max_rows).copy() if max_rows is not None else frame
    except Exception:
        return fetch_hf_mirror_rows(max_rows=max_rows)


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
