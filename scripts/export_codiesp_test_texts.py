#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

CODIESP_URL = "https://zenodo.org/records/3837305/files/codiesp.zip?download=1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(CODIESP_URL, headers={"User-Agent": "med-code-codiesp-dashboard/0.1"})
    with urllib.request.urlopen(req, timeout=180) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    rows = []
    for name in archive.namelist():
        low = "/" + name.lower().lstrip("/")
        if "/test/text_files_en/" in low and low.endswith(".txt"):
            rows.append({
                "article_id": Path(name).stem,
                "clinical_text": archive.read(name).decode("utf-8", errors="replace"),
            })
    df = pd.DataFrame(rows).sort_values("article_id")
    if len(df) != 250:
        raise RuntimeError(f"Expected 250 CodiEsp test English texts, got {len(df)}")
    df.to_csv(out / "test_texts.csv", index=False)
    print(f"exported {len(df)} test texts")


if __name__ == "__main__":
    main()
