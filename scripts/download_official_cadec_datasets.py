#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATASETS = [
    {
        "slug": "cadec_original",
        "doi": "10.4225/08/570FB102BDAD2",
        "fedora_pid": "csiro:10948",
        "expected_description": "Original CADEC with MedDRA/SNOMED CT/AMT normalisation annotations",
    },
    {
        "slug": "cadecv2",
        "doi": "10.25919/3v5b-k950",
        "fedora_pid": "csiro:62387",
        "expected_description": "CADECv2 public release (2024)",
    },
]

BASE = "https://data.csiro.au"
USER_AGENT = "MedCode-research-downloader/1.0 (+https://github.com/xm2325/med_code)"


def session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=6,
        read=6,
        status=8,
        backoff_factor=2.0,
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        respect_retry_after_header=True,
    )
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def get_json(s: requests.Session, url: str, *, timeout: int = 90) -> Any:
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def find_values(obj: Any, key_names: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in key_names:
                found.append(value)
            found.extend(find_values(value, key_names))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(find_values(value, key_names))
    return found


def extract_collection_id(metadata: dict[str, Any]) -> int:
    candidates = find_values(metadata, {"dataCollectionId", "dataCollectionID", "collectionId"})
    for value in candidates:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    for value in find_values(metadata, {"self", "data"}):
        if isinstance(value, str):
            match = re.search(r"/collections/(\d+)(?:/|$)", value)
            if match:
                return int(match.group(1))
    raise RuntimeError("Could not derive numeric CSIRO data collection ID from metadata")


def extract_download_entries(obj: Any) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            url = value.get("downloadUrl") or value.get("downloadURL") or value.get("download_url")
            if isinstance(url, str) and url.startswith("http"):
                name = (
                    value.get("filename")
                    or value.get("fileName")
                    or value.get("name")
                    or Path(urlparse(url).path).name
                    or f"file_{value.get('id', len(entries)+1)}"
                )
                if url not in seen:
                    entries.append({"filename": str(name), "download_url": url, "file_id": str(value.get("id", ""))})
                    seen.add(url)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)
    return entries


def derive_old_data_endpoint(metadata: dict[str, Any], collection_id: int) -> list[str]:
    urls: list[str] = []
    for value in find_values(metadata, {"data"}):
        if isinstance(value, str) and value.startswith("http"):
            urls.append(value)
    urls.extend([
        f"{BASE}/dap/ws/v2/collections/{collection_id}/data",
        f"{BASE}/dap/ws/v2/collections/{collection_id}/data.json",
    ])
    return list(dict.fromkeys(urls))


def discover_files(s: requests.Session, dataset_dir: Path, metadata: dict[str, Any], collection_id: int) -> list[dict[str, str]]:
    diagnostics = dataset_dir / "official_api"
    diagnostics.mkdir(parents=True, exist_ok=True)

    api_urls = [
        f"{BASE}/dap/api/v2/collections/{collection_id}/folders/contents",
        f"{BASE}/dap/api/v2/collections/{collection_id}/files/summary",
        *derive_old_data_endpoint(metadata, collection_id),
    ]
    all_entries: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for index, url in enumerate(api_urls, start=1):
        try:
            payload = get_json(s, url)
            save_json(diagnostics / f"file_api_{index}.json", {"url": url, "payload": payload})
            all_entries.extend(extract_download_entries(payload))
        except Exception as exc:  # preserve every official endpoint outcome
            errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    save_json(diagnostics / "file_api_errors.json", errors)

    dedup: dict[str, dict[str, str]] = {}
    for item in all_entries:
        dedup[item["download_url"]] = item
    entries = list(dedup.values())
    save_json(diagnostics / "discovered_files.json", entries)
    if not entries:
        raise RuntimeError(f"No downloadable CSIRO file URLs discovered for collection {collection_id}; inspect official_api diagnostics")
    return entries


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(name: str, fallback: str) -> str:
    name = os.path.basename(str(name)).strip() or fallback
    return re.sub(r"[^A-Za-z0-9._()\-]+", "_", name)


def download_file(s: requests.Session, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    with s.get(url, stream=True, timeout=(60, 300)) as response:
        response.raise_for_status()
        with temp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp.replace(destination)


def inspect_archive(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "is_zip": zipfile.is_zipfile(path),
        "archive_members": [],
        "meddra_path_hits": [],
        "annotation_path_hits": [],
        "text_path_hits": [],
    }
    if not result["is_zip"]:
        return result
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        result["archive_members"] = names
        result["meddra_path_hits"] = [n for n in names if "meddra" in n.lower()]
        result["annotation_path_hits"] = [n for n in names if n.lower().endswith((".ann", ".txt", ".csv", ".tsv", ".json")) and any(k in n.lower() for k in ("original", "annotation", "ade", "adr", "meddra"))]
        result["text_path_hits"] = [n for n in names if "/text/" in f"/{n.lower()}" or n.lower().startswith("text/")]
    return result


def process_dataset(s: requests.Session, root: Path, spec: dict[str, str]) -> dict[str, Any]:
    dataset_dir = root / spec["slug"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_url = f"{BASE}/dap/ws/v2/collections/{spec['doi']}.json"
    metadata = get_json(s, metadata_url)
    save_json(dataset_dir / "official_metadata.json", {"url": metadata_url, "payload": metadata})
    collection_id = extract_collection_id(metadata)
    entries = discover_files(s, dataset_dir, metadata, collection_id)

    downloads: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        filename = safe_filename(entry.get("filename", ""), f"download_{index}.bin")
        destination = dataset_dir / "raw" / filename
        download_file(s, entry["download_url"], destination)
        inspection = inspect_archive(destination)
        downloads.append({**entry, **inspection, "saved_path": str(destination.relative_to(root))})

    dataset_summary = {
        "slug": spec["slug"],
        "doi": spec["doi"],
        "fedora_pid": spec["fedora_pid"],
        "expected_description": spec["expected_description"],
        "numeric_collection_id": collection_id,
        "download_count": len(downloads),
        "downloads": downloads,
        "contains_meddra_named_paths": any(item["meddra_path_hits"] for item in downloads),
        "contains_annotation_named_paths": any(item["annotation_path_hits"] for item in downloads),
        "contains_text_directory_paths": any(item["text_path_hits"] for item in downloads),
    }
    save_json(dataset_dir / "dataset_manifest.json", dataset_summary)
    return dataset_summary


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "official_cadec_datasets").resolve()
    output.mkdir(parents=True, exist_ok=True)
    s = session()
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for spec in DATASETS:
        try:
            summaries.append(process_dataset(s, output, spec))
        except Exception as exc:
            failures.append({"slug": spec["slug"], "doi": spec["doi"], "error": f"{type(exc).__name__}: {exc}"})
    complete = not failures and len(summaries) == len(DATASETS)
    overall = {
        "complete": complete,
        "dataset_count": len(summaries),
        "failures": failures,
        "datasets": summaries,
        "meddra_note": "Corpus annotations may reference MedDRA concept identifiers. The licensed MedDRA terminology release itself is not downloaded or redistributed.",
    }
    save_json(output / "DOWNLOAD_MANIFEST.json", overall)
    if not complete:
        raise SystemExit(json.dumps(overall, indent=2))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
