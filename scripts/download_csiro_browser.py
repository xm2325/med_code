#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from playwright.async_api import async_playwright

DATASETS = [
    {"slug": "cadec_original", "doi": "10.4225/08/570FB102BDAD2", "expected_version": 3},
    {"slug": "cadecv2_v4", "doi": "10.25919/3v5b-k950", "expected_version": 4},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(raw: str, fallback: str) -> str:
    name = Path(urlparse(raw).path).name or fallback
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:220] or fallback


def recurse_urls(node: Any) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            out.extend(recurse_urls(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(recurse_urls(value))
    elif isinstance(node, str) and node.startswith("http"):
        if "/dap/ws/v2/collections/" in node and "/data/" in node:
            out.append(node)
        elif "s3.data.csiro.au/" in node:
            out.append(node)
    return list(dict.fromkeys(out))


def extract_file_nodes(node: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(node, dict):
        lowered = {str(key).lower(): value for key, value in node.items()}
        name = lowered.get("filename") or lowered.get("file_name") or lowered.get("name") or lowered.get("title")
        file_id = (
            lowered.get("fileid")
            or lowered.get("file_id")
            or lowered.get("datafileid")
            or lowered.get("data_file_id")
            or lowered.get("id")
        )
        download_url = lowered.get("downloadurl") or lowered.get("download_url")
        if name and file_id is not None:
            out.append({
                "name": str(name),
                "file_id": str(file_id),
                "download_url": str(download_url or ""),
                "raw": node,
            })
        for value in node.values():
            out.extend(extract_file_nodes(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(extract_file_nodes(value))
    unique = []
    seen = set()
    for item in out:
        key = (item["name"], item["file_id"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


async def browser_json(page, url: str) -> dict[str, Any]:
    return await page.evaluate(
        """async (url) => {
          const r = await fetch(url, {headers: {'Accept': 'application/json'}, credentials: 'include'});
          return {status: r.status, text: await r.text(), headers: Object.fromEntries(r.headers.entries())};
        }""",
        url,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifests = []

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        captured_urls: list[str] = []
        page.on("response", lambda response: captured_urls.append(response.url))

        for dataset in DATASETS:
            dest = root / dataset["slug"]
            dest.mkdir(parents=True, exist_ok=True)
            metadata_url = f"https://data.csiro.au/dap/ws/v2/collections/{dataset['doi']}.json"
            metadata_response = requests.get(metadata_url, timeout=120, headers={"Accept": "application/json"})
            metadata_response.raise_for_status()
            metadata = metadata_response.json()
            if int(metadata.get("versionNumber", -1)) != dataset["expected_version"]:
                raise RuntimeError(f"Unexpected version for {dataset['doi']}")
            (dest / "official_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            captured_urls.clear()
            landing = metadata["landingPage"]["href"]
            await page.goto(landing, wait_until="domcontentloaded", timeout=120000)
            await page.wait_for_timeout(5000)
            collection_id = int(metadata["dataCollectionId"])

            api_responses = {}
            endpoint_urls = {
                "files_summary": f"https://data.csiro.au/dap/api/v2/collections/{collection_id}/files/summary",
                "root_folder": f"https://data.csiro.au/dap/api/v2/collections/{collection_id}/folders/contents?folder=%2F&size=1000&page=1&q=&sb=default&so=ASC",
                "folders": f"https://data.csiro.au/dap/ws/v2/collections/{collection_id}/folders?q=",
            }
            parsed_payloads = []
            for label, endpoint in endpoint_urls.items():
                fetched = await browser_json(page, endpoint)
                api_responses[label] = {"url": endpoint, **fetched}
                if fetched["status"] == 200:
                    try:
                        parsed_payloads.append(json.loads(fetched["text"]))
                    except json.JSONDecodeError:
                        pass
            (dest / "browser_file_api_responses.json").write_text(json.dumps(api_responses, indent=2), encoding="utf-8")

            anchors = await page.locator("a").evaluate_all("els => els.map(e => e.href).filter(Boolean)")
            resources = await page.evaluate("performance.getEntriesByType('resource').map(x => x.name)")
            candidate_urls = [
                url for url in list(dict.fromkeys(captured_urls + anchors + resources))
                if re.search(r"/dap/ws/v2/collections/\d+/data/\d+(?:$|[?#])", url)
            ]
            file_nodes: list[dict[str, Any]] = []
            for payload in parsed_payloads:
                candidate_urls.extend(recurse_urls(payload))
                file_nodes.extend(extract_file_nodes(payload))
            name_by_url: dict[str, str] = {}
            for node in file_nodes:
                direct_url = node.get("download_url") or f"https://data.csiro.au/dap/ws/v2/collections/{collection_id}/data/{node['file_id']}"
                candidate_urls.append(direct_url)
                name_by_url[direct_url] = node["name"]
            candidate_urls = list(dict.fromkeys(candidate_urls))

            (dest / "browser_page_snapshot.html").write_text(await page.content(), encoding="utf-8")
            (dest / "browser_discovery.json").write_text(json.dumps({
                "landing_page": landing,
                "title": await page.title(),
                "collection_id": collection_id,
                "candidate_urls": candidate_urls,
                "file_nodes": file_nodes,
                "all_data_related_urls": [u for u in list(dict.fromkeys(captured_urls + anchors + resources)) if "/data" in u or "/folders" in u or "/files" in u],
            }, indent=2), encoding="utf-8")

            files, errors = [], []
            for index, url in enumerate(candidate_urls, start=1):
                try:
                    response = await context.request.get(url, timeout=300000)
                    if not response.ok:
                        raise RuntimeError(f"HTTP {response.status}")
                    body = await response.body()
                    disposition = response.headers.get("content-disposition", "")
                    match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', disposition, flags=re.I)
                    source_name = name_by_url.get(url, "")
                    name = safe_filename((match.group(1) if match else "") or source_name or url, f"file_{index:03d}")
                    path = dest / name
                    if path.exists():
                        path = dest / f"{index:03d}_{name}"
                    path.write_bytes(body)
                    files.append({
                        "filename": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "source_url": url,
                        "content_type": response.headers.get("content-type", ""),
                    })
                except Exception as exc:
                    errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})

            manifest = {
                **dataset,
                "landing_page": landing,
                "collection_id": collection_id,
                "candidate_count": len(candidate_urls),
                "downloaded_file_count": len(files),
                "files": files,
                "errors": errors,
            }
            (dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            manifests.append(manifest)

        await browser.close()

    (root / "download_summary.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(json.dumps(manifests, indent=2))
    if any(item["downloaded_file_count"] == 0 for item in manifests):
        raise SystemExit("Browser session did not download at least one collection; inspect browser discovery artifacts.")


if __name__ == "__main__":
    asyncio.run(main())
