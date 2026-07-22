#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortcoder.candidate_rationales import build_review_packet
from cohortcoder.knowledge import load_terminology_knowledge
from cohortcoder.uncertainty import ReviewRoutingPolicy, candidate_uncertainty, simple_ood_flag


def _parse_candidates(value: object) -> list[dict]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Create uncertainty-aware Top-K human-choice packets with grounded rationale per option")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--terminology", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coding-system")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--auto-threshold", type=float, default=0.85)
    parser.add_argument("--topk-choice-threshold", type=float, default=0.45)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions, dtype=str, keep_default_na=False)
    terminology = load_terminology_knowledge(args.terminology, coding_system=args.coding_system)
    policy = ReviewRoutingPolicy(
        auto_threshold=args.auto_threshold,
        topk_choice_threshold=args.topk_choice_threshold,
        top_k=args.top_k,
    )

    packets = []
    route_counts: dict[str, int] = {}
    for _, row in predictions.iterrows():
        candidates = _parse_candidates(row.get("candidates_json", "[]"))
        uncertainty = candidate_uncertainty(candidates)
        ood = simple_ood_flag(top_score=uncertainty.get("top_score"))
        route = policy.route(
            confidence=float(row.get("confidence", 0.0) or 0.0),
            uncertainty=uncertainty,
            explanation_gate=row.get("quality_gate") or row.get("explanation_quality_gate") or None,
            ood_flag=ood,
        )
        packet = build_review_packet(row.to_dict(), terminology, route=route, uncertainty={**uncertainty, "ood_flag": ood}, top_k=args.top_k)
        packets.append(packet)
        route_counts[route] = route_counts.get(route, 0) + 1

    with (output / "review_packets.jsonl").open("w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False) + "\n")

    flat = []
    for packet in packets:
        for option in packet["candidate_options"]:
            flat.append({
                "record_id": packet["record_id"],
                "route": packet["route"],
                "confidence": packet["confidence"],
                "candidate_rank": option["rank"],
                "candidate_code": option["code"],
                "candidate_term": option["term"],
                "model_score": option["model_score"],
                "grounded": option["grounded"],
                "evidence_quotes_json": json.dumps(option["evidence_quotes"], ensure_ascii=False),
                "rationale": option["rationale"],
                "historical_support_json": json.dumps(option["historical_support"], ensure_ascii=False),
            })
    pd.DataFrame(flat).to_csv(output / "candidate_options.csv", index=False)

    cards = []
    for packet in packets:
        options = []
        for option in packet["candidate_options"]:
            options.append(
                f"<li><b>#{option['rank']} {escape(option['code'])} — {escape(option['term'])}</b> "
                f"score={option['model_score']:.4f}<br><b>Evidence:</b> {escape('; '.join(option['evidence_quotes']) or 'No grounded exact span')}"
                f"<br><b>Why:</b> {escape(option['rationale'])}</li>"
            )
        cards.append(f"""
<section class='card'>
<h2>{escape(packet['record_id'])} — {escape(packet['route'])}</h2>
<p><b>Confidence:</b> {packet['confidence']:.4f}</p>
<div class='record'>{escape(packet['text'])}</div>
<p><b>Mention:</b> {escape(packet['mention'])}</p>
<h3>Choose from grounded candidates</h3><ol>{''.join(options)}</ol>
<p><b>Human selected code:</b> ____________________</p>
<p><b>Reason / correction note:</b> __________________________________________</p>
</section>""")
    html = """<!doctype html><html><head><meta charset='utf-8'><title>MedCode Top-K review</title>
<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}.card{border:1px solid #bbb;border-radius:10px;padding:1rem;margin:1rem 0}.record{white-space:pre-wrap;background:#fafafa;padding:.8rem}li{margin:.9rem 0}</style></head><body>
<h1>MedCode uncertainty-aware Top-K human review</h1><p>Every displayed candidate includes source-grounded evidence and a candidate-specific rationale. Missing evidence is shown explicitly rather than invented.</p>""" + "\n".join(cards) + "</body></html>"
    (output / "review_packets.html").write_text(html, encoding="utf-8")
    (output / "routing_summary.json").write_text(json.dumps({"n": len(packets), "route_counts": route_counts}, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(packets), "route_counts": route_counts}, indent=2))


if __name__ == "__main__":
    main()
