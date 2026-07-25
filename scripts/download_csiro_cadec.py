#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests

DATASETS = [
    {"slug": "cadec_original", "doi": "10.4225/08/570FB102BDAD2", "fedora_pid": "csiro:10948", "expected_version": 3, "data_gov_id": "fedora-pid_csiro-10948"},
    {"slug": "cadecv2_v4", "doi": "10.25919/3v5b-k950", "fedora_pid": "csiro:62387", "expected_version": 4, "data_gov_id": "fedora-pid_csiro-62387"},
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


def request_with_retry(session: requests.Session, url: str, *, accept_json: bool = False, timeout: int = 300) -> requests.Response:
    headers = {"Accept": "application/json"} if accept_json else {}
    last: requests.Response | None = None
    for attempt in range(7):
        response = session.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        last = response
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        retry_after = response.headers.get("Retry-After")
        delay = int(retry_after) if retry_after and retry_after.isdigit() else min(60, 5 * (2 ** attempt))
        print(f"Transient response {response.status_code} for {url}; waiting {delay}s before retry {attempt + 2}/7", flush=True)
        time.sleep(delay)
    assert last is not None
    last.raise_for_status()
    return last


def get_json(session: requests.Session, url: str) -> Any:
    return request_with_retry(session, url, accept_json=True, timeout=120).json()


def extract_file_candidates(obj: Any) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            name = str(node.get("fileName") or node.get("filename") or node.get("name") or node.get("title") or node.get("file") or "")
            for key, value in node.items():
                if isinstance(value, str) and value.startswith("http"):
                    key_lower = str(key).lower()
                    if any(token in key_lower for token in ("download", "content", "file", "href", "url", "link")):
                        candidates.append({"name": name, "url": value, "source_key": str(key)})
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
        # Only actual CSIRO collection file endpoints, not metadata/landing links.
        if "/dap/ws/v2/collections/" not in url or "/data/" not in url:
            continue
        if url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique


def data_gov_file_listing(session: requests.Session, package_id: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    url = f"https://data.gov.au/data/api/3/action/package_show?id={quote(package_id, safe='-_')}"
    payload = get_json(session, url)
    if not payload.get("success"):
        raise RuntimeError(f"data.gov.au package_show failed for {package_id}")
    result = payload.get("result") or {}
    candidates = []
    for resource in result.get("resources", []):
        resource_url = str(resource.get("url") or "")
        if resource_url.startswith("http") and "/dap/ws/v2/collections/" in resource_url and "/data/" in resource_url:
            candidates.append({"name": str(resource.get("name") or resource.get("description") or ""), "url": resource_url, "source_key": "data.gov.au:resource.url"})
    return {"api_url": url, "package": result}, candidates


def download_candidate(session: requests.Session, item: dict[str, str], destination: Path, index: int) -> dict[str, Any]:
    response = request_with_retry(session, item["url"], timeout=300)
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', disposition, flags=re.I)
    filename = safe_name((match.group(1) if match else "") or item.get("name", "") or response.url, f"file_{index:03d}")
    path = destination / filename
    if path.exists():
        path = destination / f"{index:03d}_{filename}"
    path.write_bytes(response.content)
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "content_type": response.headers.get("content-type", ""),
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
        metadata_url = f"https://data.csiro.au/dap/ws/v2/collections/{quote(dataset['doi'], safe='/')}.json"
        metadata = get_json(session, metadata_url)
        actual_version = int(metadata.get("versionNumber", -1)) if isinstance(metadata, dict) else -1
        if actual_version != int(dataset["expected_version"]):
            raise RuntimeError(f"DOI metadata version mismatch for {dataset['doi']}: expected {dataset['expected_version']}, got {actual_version}")
        (dest / "official_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        listing_source = "CSIRO DAP API"
        listing_url = str(metadata.get("data"))
        file_listing: Any = None
        file_candidates: list[dict[str, str]] = []
        try:
            file_listing = get_json(session, listing_url)
            file_candidates = extract_file_candidates(file_listing)
        except Exception as exc:
            print(f"CSIRO list endpoint unavailable for {dataset['slug']}: {type(exc).__name__}: {exc}", flush=True)

        if not file_candidates:
            listing_source = "data.gov.au harvested CSIRO resources"
            harvested, file_candidates = data_gov_file_listing(session, dataset["data_gov_id"])
            file_listing = harvested
            listing_url = harvested["api_url"]

        (dest / "official_file_listing.json").write_text(json.dumps(file_listing, indent=2, ensure_ascii=False), encoding="utf-8")
        files, errors = [], []
        for index, item in enumerate(file_candidates, start=1):
            try:
                files.append(download_candidate(session, item, dest, index))
            except Exception as exc:
                errors.append({"candidate": item, "error": f"{type(exc).__name__}: {exc}"})

        manifest = {
            **dataset,
            "actual_version": actual_version,
            "metadata_url": metadata_url,
            "listing_source": listing_source,
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
        raise SystemExit("At least one public collection produced no downloaded files; inspect listing and manifest errors.")


if __name__ == "__main__":
    main()
