#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright

DATASETS = [
    {
        "slug": "cadec_original",
        "doi": "10.4225/08/570FB102BDAD2",
        "fedora_pid": "csiro:10948",
        "landing_url": "https://data.csiro.au/collection/csiro:10948",
    },
    {
        "slug": "cadecv2",
        "doi": "10.25919/3v5b-k950",
        "fedora_pid": "csiro:62387",
        "landing_url": "https://data.csiro.au/collection/csiro:62387",
    },
]


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def walk_downloads(value: Any, out: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        url = value.get("downloadUrl") or value.get("downloadURL") or value.get("download_url")
        if isinstance(url, str) and url.startswith("http"):
            filename = (
                value.get("filename")
                or value.get("fileName")
                or value.get("name")
                or Path(urlparse(url).path).name
                or f"file_{value.get('id', len(out)+1)}"
            )
            out.append({
                "filename": str(filename),
                "download_url": url,
                "file_id": str(value.get("id", "")),
            })
        for child in value.values():
            walk_downloads(child, out)
    elif isinstance(value, list):
        for child in value:
            walk_downloads(child, out)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str, fallback: str) -> str:
    name = os.path.basename(str(name)).strip() or fallback
    return re.sub(r"[^A-Za-z0-9._()\-]+", "_", name)


def inspect_file(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "is_zip": zipfile.is_zipfile(path),
        "archive_member_count": 0,
        "archive_members": [],
        "meddra_path_hits": [],
        "text_path_hits": [],
        "annotation_path_hits": [],
    }
    if not item["is_zip"]:
        return item
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    item["archive_member_count"] = len(names)
    item["archive_members"] = names
    item["meddra_path_hits"] = [n for n in names if "meddra" in n.lower()]
    item["text_path_hits"] = [n for n in names if "/text/" in f"/{n.lower()}" or n.lower().startswith("text/")]
    item["annotation_path_hits"] = [
        n for n in names
        if n.lower().endswith((".ann", ".txt", ".csv", ".tsv", ".json"))
        and any(word in n.lower() for word in ("meddra", "annotation", "original", "ade", "adr"))
    ]
    return item


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "MedCode-official-data-downloader/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=(60, 300)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)


def capture_dataset(page, root: Path, spec: dict[str, str]) -> dict[str, Any]:
    dataset_dir = root / spec["slug"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    captured: list[dict[str, Any]] = []

    def on_response(response) -> None:
        url = response.url
        if "data.csiro.au" not in url:
            return
        if not any(token in url for token in ("/folders/contents", "/files/summary", "/data", "/collection")):
            return
        try:
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type.lower():
                return
            payload = response.json()
            captured.append({"url": url, "status": response.status, "payload": payload})
        except Exception as exc:
            captured.append({"url": url, "status": response.status, "error": f"{type(exc).__name__}: {exc}"})

    page.on("response", on_response)
    page.goto(spec["landing_url"], wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(8_000)

    # Try common tabs/buttons only when present. Page network responses are authoritative;
    # clicks simply prompt the official UI to request its file listing.
    for label in ("Data", "Files", "Download", "Access data"):
        try:
            locator = page.get_by_text(label, exact=True)
            if locator.count() > 0:
                locator.first.click(timeout=3_000)
                page.wait_for_timeout(3_000)
        except Exception:
            pass

    page.wait_for_timeout(5_000)
    (dataset_dir / "landing_page.html").write_text(page.content(), encoding="utf-8")
    page.screenshot(path=str(dataset_dir / "landing_page.png"), full_page=True)
    save_json(dataset_dir / "captured_network_responses.json", captured)

    entries: list[dict[str, str]] = []
    for record in captured:
        if "payload" in record:
            walk_downloads(record["payload"], entries)
    dedup: dict[str, dict[str, str]] = {item["download_url"]: item for item in entries}
    entries = list(dedup.values())
    save_json(dataset_dir / "captured_download_urls.json", entries)
    if not entries:
        raise RuntimeError(f"No official signed downloadUrl captured from {spec['landing_url']}")

    downloaded: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        filename = safe_name(entry["filename"], f"download_{index}.bin")
        destination = dataset_dir / "raw" / filename
        download(entry["download_url"], destination)
        downloaded.append({**entry, **inspect_file(destination), "saved_path": str(destination.relative_to(root))})

    manifest = {
        "slug": spec["slug"],
        "doi": spec["doi"],
        "fedora_pid": spec["fedora_pid"],
        "landing_url": spec["landing_url"],
        "download_count": len(downloaded),
        "downloads": downloaded,
        "contains_meddra_named_paths": any(x["meddra_path_hits"] for x in downloaded),
        "contains_text_directory_paths": any(x["text_path_hits"] for x in downloaded),
        "contains_annotation_named_paths": any(x["annotation_path_hits"] for x in downloaded),
    }
    save_json(dataset_dir / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "official_cadec_datasets").resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        for spec in DATASETS:
            page = context.new_page()
            try:
                manifests.append(capture_dataset(page, root, spec))
            except Exception as exc:
                failures.append({"slug": spec["slug"], "doi": spec["doi"], "error": f"{type(exc).__name__}: {exc}"})
            finally:
                page.close()
        browser.close()

    overall = {
        "complete": not failures and len(manifests) == len(DATASETS),
        "dataset_count": len(manifests),
        "failures": failures,
        "datasets": manifests,
        "source_boundary": "Files were downloaded from signed URLs returned by the official CSIRO Data Access Portal page network responses.",
        "meddra_boundary": "CADEC annotations may contain MedDRA identifiers; the licensed MedDRA terminology distribution itself is not included.",
    }
    save_json(root / "DOWNLOAD_MANIFEST.json", overall)
    print(json.dumps(overall, indent=2))
    if not overall["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
