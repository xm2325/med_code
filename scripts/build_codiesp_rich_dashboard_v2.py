#!/usr/bin/env python3
"""Materialise the archived CodiEsp first-10 rich dashboard v2.

This utility is intentionally OFFLINE: it makes no DeepSeek/LLM/API calls and does
not run an experiment. The generated dashboard was built from the already-completed
first-10 Direct vs ICD-Knowledge-RAG shard outputs. To keep the large generated HTML
manageable in GitHub, the exact HTML is stored as gzip+base64 payload parts under
``reports/``. This script reconstructs that exact dashboard byte-for-byte.

Usage:
    python scripts/build_codiesp_rich_dashboard_v2.py
    python scripts/build_codiesp_rich_dashboard_v2.py --check-only
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
from pathlib import Path


PART_NAMES = (
    "codiesp_first10_rich_dashboard_v2.payload.b64.part01",
    "codiesp_first10_rich_dashboard_v2.payload.b64.part02",
    "codiesp_first10_rich_dashboard_v2.payload.b64.part03",
)
DEFAULT_OUTPUT = "codiesp_first10_rich_dashboard_v2.full.html"


def materialise(reports_dir: Path) -> bytes:
    """Read payload parts and return the exact decompressed HTML bytes."""
    chunks: list[str] = []
    for name in PART_NAMES:
        path = reports_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing dashboard payload part: {path}")
        chunks.append(path.read_text(encoding="utf-8"))

    packed_b64 = "".join(chunks)
    packed_b64 = "".join(packed_b64.split())
    compressed = base64.b64decode(packed_b64, validate=True)
    return gzip.decompress(compressed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports",
        help="Directory containing the three archived payload parts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output HTML path (default: reports/{DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate/decompress the payload and print metadata without writing a file.",
    )
    args = parser.parse_args()

    html_bytes = materialise(args.reports_dir)
    digest = hashlib.sha256(html_bytes).hexdigest()

    if not args.check_only:
        output = args.output or (args.reports_dir / DEFAULT_OUTPUT)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(html_bytes)
        print(f"Wrote {output}")

    print(f"HTML bytes: {len(html_bytes):,}")
    print(f"SHA256: {digest}")
    print("No LLM/API calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
