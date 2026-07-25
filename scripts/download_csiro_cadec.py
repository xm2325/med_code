#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests

DATASETS = [
    {
        "slug": "cadec_original",
        "doi": "10.4225/08/570FB102BDAD2",
        "fedora_pid": "csiro:10948",
    },
    {
        "slug": "cadecv2_v4",
        "doi": "10.25919/3v5b-k950",
        "fedora_pid": "csiro:62387v4",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: str, fallback: str) -> str:
    value = Path(urlparse(value).path).name or fallback
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value[:220] or fallback


def get_json(session: requests.Session, url: str) -> Any:
    response = session.get(url, timeout=120, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def extract_file_candidates(obj: Any) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            name = str(
                node.get("fileName")
                or node.get("filename")
                or node.get("name")
                or node.get("title")
                or ""
            )
            preferred_keys = [
                "downloadURL", "downloadUrl", "download_url", "contentUrl",
                "fileUrl", "fileURL", "href", "url", "self",
            ]
            for key in preferred_keys:
                value = node.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    lower = value.lower()
                    if any(token in key.lower() for token in ("download", "content", "file")) or "/data/" in lower:
                        candidates.append({"name": name, "url": value, "source_key": key})
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(obj)
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in candidates:
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique


def download_candidate(session: requests.Session, item: dict[str, str], destination: Path, index: int) -> dict[str, Any]:
    response = session.get(item["url"], timeout=300, allow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    disposition = response.headers.get("content-disposition", "")
    disposition_name = ""
    match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', disposition, flags=re.I)
    if match:
        disposition_name = match.group(1)
    filename = safe_name(disposition_name or item.get("name", "") or response.url, f"file_{index:03d}")
    path = destination / filename
    if path.exists():
        path = destination / f"{index:03d}_{filename}"
    path.write_bytes(response.content)
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "content_type": content_type,
        "source_url": item["url"],
        "resolved_url": response.url,
        "source_key": item.get("source_key", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "MedCode-research-downloader/0.2.2"})
    overall = []

    for dataset in DATASETS:
        dest = root / dataset["slug"]
        dest.mkdir(parents=True, exist_ok=True)
        doi = dataset["doi"]
        metadata_url = f"https://data.csiro.au/dap/ws/v2/collections/{quote(doi, safe='/')}.json"
        metadata = get_json(session, metadata_url)
        (dest / "official_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        data_url = metadata.get("data") if isinstance(metadata, dict) else None
        if not data_url:
            data_url = f"https://data.csiro.au/dap/ws/v2/collections/{quote(dataset['fedora_pid'], safe=':')}/data"
        listing_url = str(data_url)
        file_listing = get_json(session, listing_url)
        (dest / "official_file_listing.json").write_text(json.dumps(file_listing, indent=2, ensure_ascii=False), encoding="utf-8")

        file_candidates = extract_file_candidates(file_listing)
        files = []
        errors = []
        for index, item in enumerate(file_candidates, start=1):
            try:
                result = download_candidate(session, item, dest, index)
                files.append(result)
            except Exception as exc:
                errors.append({"candidate": item, "error": f"{type(exc).__name__}: {exc}"})

        manifest = {
            **dataset,
            "metadata_url": metadata_url,
            "data_listing_url": listing_url,
            "candidate_count": len(file_candidates),
            "downloaded_file_count": len(files),
            "files": files,
            "errors": errors,
        }
        (dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        overall.append(manifest)

    (root / "download_summary.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    if any(item["downloaded_file_count"] == 0 for item in overall):
        raise SystemExit("At least one public CSIRO collection produced no downloaded files; inspect official_file_listing.json and manifest errors.")


if __name__ == "__main__":
    main()
