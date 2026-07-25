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
    elif isinstance(node, str) and node.startswith("http") and "/dap/ws/v2/collections/" in node and "/data/" in node:
        out.append(node)
    return list(dict.fromkeys(out))


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
            await page.wait_for_timeout(8000)
            # Attempt to expose lazy-loaded data/file panels without relying on one UI version.
            for pattern in [re.compile("files", re.I), re.compile("data", re.I), re.compile("download", re.I)]:
                try:
                    locator = page.get_by_text(pattern).first
                    if await locator.count():
                        await locator.click(timeout=3000)
                        await page.wait_for_timeout(3000)
                except Exception:
                    pass

            anchors = await page.locator("a").evaluate_all("els => els.map(e => e.href).filter(Boolean)")
            resources = await page.evaluate("performance.getEntriesByType('resource').map(x => x.name)")
            candidate_urls = [
                url for url in list(dict.fromkeys(captured_urls + anchors + resources))
                if "/dap/ws/v2/collections/" in url and "/data/" in url
            ]

            data_url = metadata.get("data")
            listing_payload: Any = None
            if data_url:
                fetched = await page.evaluate(
                    """async (url) => {
                      const r = await fetch(url, {headers: {'Accept': 'application/json'}, credentials: 'include'});
                      return {status: r.status, text: await r.text(), headers: Object.fromEntries(r.headers.entries())};
                    }""",
                    data_url,
                )
                (dest / "browser_data_response.json").write_text(json.dumps(fetched, indent=2), encoding="utf-8")
                if fetched["status"] == 200:
                    try:
                        listing_payload = json.loads(fetched["text"])
                        candidate_urls.extend(recurse_urls(listing_payload))
                    except json.JSONDecodeError:
                        pass
            candidate_urls = list(dict.fromkeys(url for url in candidate_urls if re.search(r"/data/\d+(?:$|[?#])", url)))

            (dest / "browser_page_snapshot.html").write_text(await page.content(), encoding="utf-8")
            (dest / "browser_discovery.json").write_text(json.dumps({
                "landing_page": landing,
                "title": await page.title(),
                "candidate_urls": candidate_urls,
                "all_data_related_urls": [u for u in list(dict.fromkeys(captured_urls + anchors + resources)) if "/data" in u],
                "listing_payload": listing_payload,
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
                    name = safe_filename((match.group(1) if match else "") or url, f"file_{index:03d}")
                    path = dest / name
                    path.write_bytes(body)
                    files.append({"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path), "source_url": url, "content_type": response.headers.get("content-type", "")})
                except Exception as exc:
                    errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})

            manifest = {**dataset, "landing_page": landing, "candidate_count": len(candidate_urls), "downloaded_file_count": len(files), "files": files, "errors": errors}
            (dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            manifests.append(manifest)

        await browser.close()

    (root / "download_summary.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(json.dumps(manifests, indent=2))
    if any(item["downloaded_file_count"] == 0 for item in manifests):
        raise SystemExit("Browser session did not download at least one collection; inspect browser discovery artifacts.")


if __name__ == "__main__":
    asyncio.run(main())
