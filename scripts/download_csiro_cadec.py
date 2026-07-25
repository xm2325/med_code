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
    {"slug": "cadec_original", "doi": "10.4225/08/570FB102BDAD2", "fedora_pid": "csiro:10948", "expected_version": 3},
    {"slug": "cadecv2_v4", "doi": "10.25919/3v5b-k950", "fedora_pid": "csiro:62387", "expected_version": 4},
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
    for attempt in range(5):
        response = session.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        last = response
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        retry_after = response.headers.get("Retry-After")
        delay = int(retry_after) if retry_after and retry_after.isdigit() else min(30, 3 * (2 ** attempt))
        print(f"Transient response {response.status_code} for {url}; waiting {delay}s before retry {attempt + 2}/5", flush=True)
        time.sleep(delay)
    assert last is not None
    last.raise_for_status()
    return last


def get_json(session: requests.Session, url: str) -> Any:
    return request_with_retry(session, url, accept_json=True, timeout=120).json()


def candidate_resources(package: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    for resource in package.get("resources", []):
        resource_url = str(resource.get("url") or "")
        if resource_url.startswith("http") and "/dap/ws/v2/collections/" in resource_url and "/data/" in resource_url:
            out.append({
                "name": str(resource.get("name") or resource.get("description") or ""),
                "url": resource_url,
                "source_key": "data.gov.au:resource.url",
            })
    return out


def data_gov_file_listing(session: requests.Session, *, doi: str, fedora_pid: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    queries = [doi, fedora_pid, "CADECv2" if "62387" in fedora_pid else "CADEC"]
    attempts = []
    for query in queries:
        url = f"https://data.gov.au/data/api/3/action/package_search?q={quote(query)}&rows=50"
        payload = get_json(session, url)
        results = ((payload.get("result") or {}).get("results") or []) if payload.get("success") else []
        attempts.append({"query": query, "api_url": url, "n_results": len(results), "package_names": [x.get("name") for x in results]})
        for package in results:
            searchable = json.dumps(package, ensure_ascii=False).lower()
            if doi.lower() not in searchable and fedora_pid.lower() not in searchable and str(package.get("title", "")).lower() not in {"cadec", "cadecv2"}:
                continue
            candidates = candidate_resources(package)
            if candidates:
                return {"api_url": url, "selected_package": package, "search_attempts": attempts}, candidates
    raise RuntimeError("No harvested CSIRO file resources found in data.gov.au searches: " + json.dumps(attempts))


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

        listing, file_candidates = data_gov_file_listing(session, doi=dataset["doi"], fedora_pid=dataset["fedora_pid"])
        (dest / "official_file_listing.json").write_text(json.dumps(listing, indent=2, ensure_ascii=False), encoding="utf-8")

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
            "listing_source": "data.gov.au harvested CSIRO resources",
            "data_listing_url": listing["api_url"],
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
