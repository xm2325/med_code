from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd


def _safe_json(value: object) -> str:
    try:
        return json.dumps(json.loads(str(value)), ensure_ascii=False, indent=2)
    except Exception:
        return str(value or "")


def build_review_casebook(
    predictions: pd.DataFrame,
    explanations: pd.DataFrame | None = None,
    *,
    max_cases: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Create an audit-first case sample for clinical/coder review.

    Priority cases are selected before random correct controls. The function assumes
    explanations were generated in the same row order as predictions; if row counts do
    not match, explanation columns are not positionally attached.
    """
    df = predictions.reset_index(drop=True).copy().fillna("")
    if explanations is not None and len(explanations) == len(df):
        exp = explanations.reset_index(drop=True).copy().fillna("")
        for column in ["why", "explanation_status", "evidence_quotes_json", "evidence_spans_json", "coding_system"]:
            if column in exp:
                df[column] = exp[column]

    correct = pd.to_numeric(df.get("correct", 0), errors="coerce").fillna(0).astype(int)
    confidence = pd.to_numeric(df.get("confidence", 0.0), errors="coerce").fillna(0.0)
    decision = df.get("decision", pd.Series("", index=df.index)).astype(str)
    novelty = df.get("code_novelty", pd.Series("", index=df.index)).astype(str)
    grounding = df.get("explanation_status", pd.Series("", index=df.index)).astype(str)

    reason = pd.Series("", index=df.index, dtype=str)
    reason.loc[(decision == "AUTO_CANDIDATE") & (correct == 0)] = "auto_candidate_error"
    reason.loc[(correct == 0) & (confidence >= confidence.quantile(0.75)) & (reason == "")] = "high_confidence_error"
    reason.loc[(correct == 0) & (novelty == "unseen_code") & (reason == "")] = "unseen_code_error"
    reason.loc[grounding.isin(["insufficient_grounding", "insufficient_affirmed_evidence"]) & (reason == "")] = "insufficient_evidence"
    reason.loc[(correct == 0) & (reason == "")] = "other_error"
    reason.loc[(correct == 1) & (reason == "")] = "correct_control"
    df["review_reason"] = reason

    priority_order = [
        "auto_candidate_error",
        "high_confidence_error",
        "unseen_code_error",
        "insufficient_evidence",
        "other_error",
    ]
    selected_parts = []
    remaining_slots = int(max_cases)
    for label in priority_order:
        subset = df[df.review_reason == label].copy()
        if subset.empty or remaining_slots <= 0:
            continue
        subset = subset.sort_values("confidence", ascending=False) if "confidence" in subset else subset
        take = min(len(subset), max(1, int(max_cases) // max(1, len(priority_order))))
        take = min(take, remaining_slots)
        selected_parts.append(subset.head(take))
        remaining_slots -= take

    already = set()
    if selected_parts:
        for part in selected_parts:
            already.update(part.index.tolist())
    if remaining_slots > 0:
        controls = df[(df.review_reason == "correct_control") & (~df.index.isin(already))]
        if len(controls) > remaining_slots:
            controls = controls.sample(n=remaining_slots, random_state=seed)
        selected_parts.append(controls)

    selected = pd.concat(selected_parts, axis=0) if selected_parts else df.head(0)
    selected = selected.head(int(max_cases)).copy()
    selected.insert(0, "casebook_id", [f"CASE-{idx+1:03d}" for idx in range(len(selected))])
    for column in ["expert_code_correct", "expert_evidence_complete", "expert_rationale_correct", "expert_action", "expert_comments"]:
        selected[column] = ""
    return selected.reset_index(drop=True)


def write_review_casebook(output_dir: str | Path, casebook: pd.DataFrame) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    casebook.to_csv(output / "review_casebook.csv", index=False)

    cards: list[str] = []
    for _, row in casebook.iterrows():
        evidence = _safe_json(row.get("evidence_quotes_json", ""))
        candidates = _safe_json(row.get("candidates_json", ""))
        context = str(row.get("text", ""))
        mention = str(row.get("mention", ""))
        cards.append(f"""
<section class='card'>
<h2>{escape(str(row.get('casebook_id', '')))} — {escape(str(row.get('review_reason', '')))}</h2>
<p><b>Record:</b> {escape(str(row.get('record_id', '')))}</p>
<p><b>Gold:</b> {escape(str(row.get('gold_code', '')))} {escape(str(row.get('gold_term', '')))}</p>
<p><b>Predicted:</b> {escape(str(row.get('predicted_code', '')))} {escape(str(row.get('predicted_term', '')))}</p>
<p><b>Confidence:</b> {escape(str(row.get('confidence', '')))} &nbsp; <b>Decision:</b> {escape(str(row.get('decision', '')))}</p>
<h3>Original context</h3><div class='record'>{escape(context)}</div>
<h3>Task mention</h3><p>{escape(mention)}</p>
<h3>Why</h3><p>{escape(str(row.get('why', '')))}</p>
<h3>Evidence</h3><pre>{escape(evidence)}</pre>
<h3>Top candidates</h3><pre>{escape(candidates)}</pre>
</section>""")
    html = """<!doctype html><html><head><meta charset='utf-8'><title>MedCode clinical review casebook</title>
<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.45}.card{border:1px solid #ccc;border-radius:10px;padding:1rem 1.3rem;margin:1.2rem 0}.record{white-space:pre-wrap;background:#fbfbfb;border:1px solid #ddd;padding:.8rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f7f7;padding:.8rem}</style>
</head><body><h1>MedCode clinical review casebook</h1><p>Priority errors and evidence failures are sampled before correct controls. This file is for expert review, not a performance summary.</p>""" + "\n".join(cards) + "</body></html>"
    (output / "review_casebook.html").write_text(html, encoding="utf-8")
