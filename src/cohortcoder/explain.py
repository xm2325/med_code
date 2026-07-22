from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
import json
import re
from typing import Any, Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class EvidenceSpan:
    start: int
    end: int
    quote: str
    source: str
    score: float
    matched_phrases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matched_phrases"] = list(self.matched_phrases)
        return payload


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", str(text)) if len(token) >= 2}


def _phrases(term: str = "", synonyms: str = "") -> list[str]:
    values = [str(term).strip()]
    values.extend(part.strip() for part in re.split(r"[|;\n]", str(synonyms)) if part.strip())
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _find_case_insensitive(text: str, needle: str) -> tuple[int, int] | None:
    if not needle.strip():
        return None
    start = text.lower().find(needle.lower())
    return None if start < 0 else (start, start + len(needle))


def _sentence_spans(text: str) -> Iterable[tuple[int, int, str]]:
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        raw = match.group(0)
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right <= left:
            continue
        start = match.start() + left
        end = match.start() + right
        yield start, end, text[start:end]


def extract_evidence_spans(
    text: str,
    *,
    mention: str = "",
    term: str = "",
    synonyms: str = "",
    max_spans: int = 3,
) -> list[EvidenceSpan]:
    """Extract verbatim evidence spans with stable character offsets.

    The explicitly supplied mention is preferred. Additional sentences are ranked by
    overlap with the mention and terminology phrases. The function never invents a
    quote: every returned span is sliced directly from ``text``.
    """
    text = str(text or "")
    mention = str(mention or "").strip()
    phrases = _phrases(term, synonyms)
    target_tokens = _tokens(" ".join([mention, *phrases]))
    spans: list[EvidenceSpan] = []

    def overlaps(start: int, end: int) -> bool:
        return any(not (end <= span.start or start >= span.end) for span in spans)

    exact = _find_case_insensitive(text, mention)
    if exact is not None:
        start, end = exact
        quote = text[start:end]
        matched = tuple(phrase for phrase in phrases if phrase.lower() in quote.lower())
        spans.append(EvidenceSpan(start, end, quote, "explicit_mention", 2.0, matched))

    candidates: list[EvidenceSpan] = []
    for start, end, sentence in _sentence_spans(text):
        if overlaps(start, end):
            continue
        sentence_tokens = _tokens(sentence)
        overlap = len(sentence_tokens & target_tokens)
        phrase_hits = tuple(phrase for phrase in phrases if phrase.lower() in sentence.lower())
        mention_bonus = 1.0 if mention and mention.lower() in sentence.lower() else 0.0
        score = (overlap / max(1, len(target_tokens))) + 0.5 * len(phrase_hits) + mention_bonus
        if score > 0:
            candidates.append(EvidenceSpan(start, end, sentence, "supporting_sentence", float(score), phrase_hits))

    candidates.sort(key=lambda span: (-span.score, span.start))
    for candidate in candidates:
        if len(spans) >= max_spans:
            break
        if not overlaps(candidate.start, candidate.end):
            spans.append(candidate)

    spans.sort(key=lambda span: span.start)
    return spans[:max_spans]


def remove_spans(text: str, spans: Iterable[EvidenceSpan]) -> str:
    chars = list(str(text))
    for span in spans:
        start = max(0, int(span.start))
        end = min(len(chars), int(span.end))
        for index in range(start, end):
            chars[index] = " "
    return "".join(chars)


def _parse_json_list(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    try:
        payload = json.loads(str(value or "[]"))
    except (json.JSONDecodeError, TypeError):
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _terminology_row(terminology: pd.DataFrame, code: str) -> Mapping[str, Any]:
    matched = terminology[terminology["code"].astype(str) == str(code)]
    return {} if matched.empty else matched.iloc[0].to_dict()


def _historical_support(value: object, code: str, limit: int = 2) -> list[dict[str, Any]]:
    rows = []
    for item in _parse_json_list(value):
        if str(item.get("code", "")) != str(code):
            continue
        rows.append({
            "text": str(item.get("text", "")),
            "code": str(item.get("code", "")),
            "term": str(item.get("term", "")),
            "similarity": float(item.get("similarity", 0.0) or 0.0),
        })
        if len(rows) >= limit:
            break
    return rows


def build_explanation_record(
    prediction: Mapping[str, Any],
    terminology: pd.DataFrame,
    *,
    coder: Any | None = None,
) -> dict[str, Any]:
    """Build an auditable explanation for one already-selected code.

    Explanation is downstream of code selection: it cannot change the predicted code.
    Faithfulness uses the exact model input (mention when present, otherwise full text).
    """
    text = str(prediction.get("text", "") or "")
    mention = str(prediction.get("mention", "") or "").strip()
    code = str(prediction.get("predicted_code", ""))
    term_row = _terminology_row(terminology, code)
    term = str(term_row.get("term", prediction.get("predicted_term", "")) or "")
    synonyms = str(term_row.get("synonyms", "") or "")
    definition = str(term_row.get("definition", "") or "")
    hierarchy = str(term_row.get("hierarchy", "") or "")
    coding_system = str(term_row.get("system", prediction.get("coding_system", "")) or "unspecified")

    display_spans = extract_evidence_spans(text, mention=mention, term=term, synonyms=synonyms)
    model_input = mention if mention else text
    model_spans = extract_evidence_spans(model_input, mention=mention, term=term, synonyms=synonyms)
    evidence_quotes = [span.quote for span in display_spans]
    matched_phrases = sorted({phrase for span in display_spans for phrase in span.matched_phrases})
    historical = _historical_support(prediction.get("historical_cases_json", "[]"), code)

    if evidence_quotes:
        quoted = "; ".join(f"“{quote}”" for quote in evidence_quotes[:2])
        why = f"The proposed {coding_system} code {code} ({term}) is grounded in the record text {quoted}."
        if matched_phrases:
            why += " The highlighted wording overlaps the terminology expression(s): " + ", ".join(matched_phrases[:4]) + "."
        if historical:
            why += " A similar historical expert-coded example with the same code was also retrieved as supporting provenance."
        status = "grounded"
    else:
        why = f"The model proposed {coding_system} code {code} ({term}), but no exact supporting span could be grounded in the supplied text. Expert review is recommended."
        status = "insufficient_grounding"

    original_score = evidence_only_score = evidence_removed_score = None
    sufficiency_gap = comprehensiveness_drop = None
    if coder is not None and hasattr(coder, "score_code") and code:
        original_score = float(coder.score_code(model_input, code))
        evidence_only_text = " ".join(span.quote for span in model_spans)
        evidence_removed_text = remove_spans(model_input, model_spans)
        evidence_only_score = float(coder.score_code(evidence_only_text, code)) if evidence_only_text.strip() else 0.0
        evidence_removed_score = float(coder.score_code(evidence_removed_text, code)) if evidence_removed_text.strip() else 0.0
        sufficiency_gap = original_score - evidence_only_score
        comprehensiveness_drop = original_score - evidence_removed_score

    return {
        "record_id": str(prediction.get("record_id", "")),
        "coding_system": coding_system,
        "predicted_code": code,
        "predicted_term": term,
        "confidence": prediction.get("confidence"),
        "decision": prediction.get("decision", ""),
        "why": why,
        "evidence_spans": [span.to_dict() for span in display_spans],
        "evidence_quotes": evidence_quotes,
        "matched_terminology_phrases": matched_phrases,
        "external_knowledge": {
            "term": term,
            "synonyms": _phrases("", synonyms),
            "definition": definition,
            "hierarchy": hierarchy,
        },
        "historical_support": historical,
        "explanation_source": "grounded_deterministic",
        "explanation_status": status,
        "evidence_verbatim": bool(evidence_quotes) and all(quote in text for quote in evidence_quotes),
        "faithfulness": {
            "model_input": "mention" if mention else "full_text",
            "original_code_score": original_score,
            "evidence_only_code_score": evidence_only_score,
            "evidence_removed_code_score": evidence_removed_score,
            "sufficiency_gap": sufficiency_gap,
            "comprehensiveness_drop": comprehensiveness_drop,
            "interpretation": "Lower sufficiency gap and higher comprehensiveness drop indicate stronger model-centric faithfulness.",
        },
        "text": text,
        "mention": mention,
    }


def explain_predictions(
    predictions: pd.DataFrame,
    terminology: pd.DataFrame,
    *,
    coder: Any | None = None,
) -> list[dict[str, Any]]:
    return [build_explanation_record(row.to_dict(), terminology, coder=coder) for _, row in predictions.iterrows()]


def _highlight(text: str, spans: Iterable[Mapping[str, Any]]) -> str:
    safe_spans = []
    for span in spans:
        try:
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start < end <= len(text):
            safe_spans.append((start, end))
    safe_spans.sort()
    parts: list[str] = []
    cursor = 0
    for start, end in safe_spans:
        if start < cursor:
            continue
        parts.append(escape(text[cursor:start]))
        parts.append("<mark>" + escape(text[start:end]) + "</mark>")
        cursor = end
    parts.append(escape(text[cursor:]))
    return "".join(parts)


def write_explanation_artifacts(output_dir: str | Path, explanations: list[dict[str, Any]]) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    flat_rows = []
    for item in explanations:
        flat_rows.append({
            "record_id": item["record_id"],
            "coding_system": item["coding_system"],
            "predicted_code": item["predicted_code"],
            "predicted_term": item["predicted_term"],
            "confidence": item["confidence"],
            "decision": item["decision"],
            "why": item["why"],
            "explanation_source": item["explanation_source"],
            "explanation_status": item["explanation_status"],
            "evidence_verbatim": item["evidence_verbatim"],
            "evidence_quotes_json": json.dumps(item["evidence_quotes"], ensure_ascii=False),
            "evidence_spans_json": json.dumps(item["evidence_spans"], ensure_ascii=False),
            "matched_terminology_phrases_json": json.dumps(item["matched_terminology_phrases"], ensure_ascii=False),
            "external_knowledge_json": json.dumps(item["external_knowledge"], ensure_ascii=False),
            "historical_support_json": json.dumps(item["historical_support"], ensure_ascii=False),
            "sufficiency_gap": item["faithfulness"]["sufficiency_gap"],
            "comprehensiveness_drop": item["faithfulness"]["comprehensiveness_drop"],
        })
    frame = pd.DataFrame(flat_rows)
    frame.to_csv(output / "explanations.csv", index=False)
    with (output / "explanations.jsonl").open("w", encoding="utf-8") as handle:
        for item in explanations:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    grounded = [item for item in explanations if item["explanation_status"] == "grounded"]
    verbatim = [item for item in explanations if item["evidence_verbatim"]]
    suff = [item["faithfulness"]["sufficiency_gap"] for item in explanations if item["faithfulness"]["sufficiency_gap"] is not None]
    comp = [item["faithfulness"]["comprehensiveness_drop"] for item in explanations if item["faithfulness"]["comprehensiveness_drop"] is not None]
    metrics = {
        "n": len(explanations),
        "grounded_rate": len(grounded) / len(explanations) if explanations else 0.0,
        "verbatim_evidence_rate": len(verbatim) / len(explanations) if explanations else 0.0,
        "mean_sufficiency_gap": sum(suff) / len(suff) if suff else None,
        "mean_comprehensiveness_drop": sum(comp) / len(comp) if comp else None,
    }
    (output / "explainability_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    review = frame[["record_id", "coding_system", "predicted_code", "predicted_term", "why", "evidence_quotes_json"]].copy() if not frame.empty else pd.DataFrame()
    for column in ["expert_code_supported", "expert_evidence_complete", "expert_rationale_correct", "expert_comments"]:
        review[column] = ""
    review.to_csv(output / "rationale_review_template.csv", index=False)

    cards = []
    for item in explanations:
        support = "".join(
            f"<li>{escape(str(case['term']))} ({escape(str(case['code']))}), similarity={case['similarity']:.3f}</li>"
            for case in item["historical_support"]
        ) or "<li>None retrieved with the same code.</li>"
        cards.append(f"""
        <section class='card'>
          <h2>{escape(item['coding_system'])} {escape(item['predicted_code'])} — {escape(item['predicted_term'])}</h2>
          <p><b>Decision:</b> {escape(str(item['decision']))} &nbsp; <b>Confidence:</b> {escape(str(item['confidence']))}</p>
          <h3>Why this code?</h3><p>{escape(item['why'])}</p>
          <h3>Evidence in the record</h3><div class='record'>{_highlight(item['text'], item['evidence_spans'])}</div>
          <h3>External terminology support</h3><pre>{escape(json.dumps(item['external_knowledge'], ensure_ascii=False, indent=2))}</pre>
          <h3>Historical expert-coded support</h3><ul>{support}</ul>
          <h3>Faithfulness audit</h3><pre>{escape(json.dumps(item['faithfulness'], ensure_ascii=False, indent=2))}</pre>
        </section>""")
    html = """<!doctype html><html><head><meta charset='utf-8'><title>MedCode explanations</title>
    <style>body{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5}.card{border:1px solid #bbb;border-radius:10px;padding:1rem 1.3rem;margin:1.2rem 0}.record{white-space:pre-wrap;padding:1rem;border:1px solid #ddd;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
    </head><body><h1>MedCode explainable coding review</h1>""" + "\n".join(cards) + "</body></html>"
    (output / "explanations.html").write_text(html, encoding="utf-8")
    return metrics
