#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from cohortcoder.adr_coding import OneLineADRCoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map one short RA treatment adverse-event record to the versioned demo MedDRA subset."
    )
    parser.add_argument("text", help="One non-empty healthcare-professional record")
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON result")
    args = parser.parse_args()

    result = OneLineADRCoder().map_line(args.text, record_id=args.record_id)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
