from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_code_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


@dataclass
class MultiLabelPrediction:
    candidates: list[dict]

    def codes_at_k(self, k: int) -> list[str]:
        return [str(item["code"]) for item in self.candidates[: int(k)]]


class MultiLabelHistoricalCoder:
    """Transparent multi-label ICD ranking baseline for long clinical documents.

    Terminology and historical-document retrieval use separate vector spaces. Historical
    evidence for a code is the maximum similarity to a TRAIN document carrying that code.
    This is an auditable baseline, not a claim of state-of-the-art long-document coding.
    """

    def __init__(self, history_weight: float = 0.5, top_k: int = 100):
        if not 0 <= history_weight <= 1:
            raise ValueError("history_weight must be between 0 and 1")
        self.history_weight = float(history_weight)
        self.top_k = int(top_k)

    @staticmethod
    def _safe_texts(values: Iterable[object]) -> list[str]:
        out = []
        for value in values:
            text = str(value or "").strip()
            out.append(text if text else "__empty__")
        return out

    def fit(self, history: pd.DataFrame, terminology: pd.DataFrame) -> "MultiLabelHistoricalCoder":
        if not {"text", "gold_codes_json"}.issubset(history.columns):
            raise ValueError("history requires text and gold_codes_json")
        if not {"code", "term"}.issubset(terminology.columns):
            raise ValueError("terminology requires code and term")
        if history.empty or terminology.empty:
            raise ValueError("history and terminology must be non-empty")

        self.history = history.reset_index(drop=True).copy()
        self.terminology = terminology.drop_duplicates("code").reset_index(drop=True).copy()
        term_text_col = "search_text" if "search_text" in self.terminology.columns else "term"
        term_texts = self._safe_texts(self.terminology[term_text_col])
        history_texts = self._safe_texts(self.history["text"])

        self.term_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.history_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.term_matrix = self.term_vectorizer.fit_transform(term_texts)
        self.history_matrix = self.history_vectorizer.fit_transform(history_texts)
        self.history_codes = [parse_code_list(value) for value in self.history["gold_codes_json"]]
        self.code_to_term = dict(zip(self.terminology["code"].astype(str), self.terminology["term"].astype(str)))
        return self

    def rank_one(self, text: str) -> MultiLabelPrediction:
        query_text = str(text or "").strip() or "__empty__"
        term_scores = cosine_similarity(self.term_vectorizer.transform([query_text]), self.term_matrix)[0]
        hist_scores = cosine_similarity(self.history_vectorizer.transform([query_text]), self.history_matrix)[0]

        history_by_code: dict[str, float] = {}
        for idx, similarity in enumerate(hist_scores):
            for code in self.history_codes[idx]:
                history_by_code[code] = max(history_by_code.get(code, 0.0), float(similarity))

        candidates: list[dict] = []
        for idx, row in self.terminology.iterrows():
            code = str(row["code"])
            terminology_score = float(term_scores[idx])
            history_score = float(history_by_code.get(code, 0.0))
            score = (1.0 - self.history_weight) * terminology_score + self.history_weight * history_score
            candidates.append({
                "code": code,
                "term": str(row["term"]),
                "score": float(score),
                "terminology_score": terminology_score,
                "history_score": history_score,
            })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return MultiLabelPrediction(candidates=candidates[: self.top_k])

    def rank(self, texts: Iterable[str]) -> list[MultiLabelPrediction]:
        return [self.rank_one(str(text)) for text in texts]


def rank_dataframe(coder: MultiLabelHistoricalCoder, records: pd.DataFrame) -> pd.DataFrame:
    if not {"record_id", "text", "gold_codes_json"}.issubset(records.columns):
        raise ValueError("records require record_id, text, gold_codes_json")
    predictions = coder.rank(records["text"].astype(str))
    rows = []
    for (_, record), prediction in zip(records.iterrows(), predictions):
        rows.append({
            "record_id": str(record["record_id"]),
            "subject_id": str(record.get("subject_id", "")),
            "gold_codes_json": str(record["gold_codes_json"]),
            "candidates_json": json.dumps(prediction.candidates),
        })
    return pd.DataFrame(rows)


def _candidate_list(value: object) -> list[dict]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def ranking_metrics(predictions: pd.DataFrame, k_values: Iterable[int] = (5, 10, 20)) -> dict[str, float]:
    result: dict[str, float] = {"n_notes": float(len(predictions))}
    if predictions.empty:
        return result
    for k in k_values:
        per_precision = []
        per_recall = []
        for _, row in predictions.iterrows():
            gold = set(parse_code_list(row["gold_codes_json"]))
            pred = [str(item.get("code", "")) for item in _candidate_list(row["candidates_json"])[: int(k)]]
            hits = len(gold & set(pred))
            per_precision.append(hits / max(1, len(pred)))
            per_recall.append(hits / max(1, len(gold)))
        result[f"precision_at_{k}"] = float(np.mean(per_precision))
        result[f"recall_at_{k}"] = float(np.mean(per_recall))
    return result


def threshold_metrics(predictions: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    tp = fp = fn = 0
    note_f1: list[float] = []
    exact = 0
    proposed = 0
    for _, row in predictions.iterrows():
        gold = set(parse_code_list(row["gold_codes_json"]))
        pred = {
            str(item.get("code", ""))
            for item in _candidate_list(row["candidates_json"])
            if float(item.get("score", 0.0) or 0.0) >= float(threshold)
        }
        pred.discard("")
        note_tp = len(gold & pred)
        note_fp = len(pred - gold)
        note_fn = len(gold - pred)
        tp += note_tp
        fp += note_fp
        fn += note_fn
        proposed += len(pred)
        precision = note_tp / max(1, note_tp + note_fp)
        recall = note_tp / max(1, note_tp + note_fn)
        note_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        exact += int(pred == gold)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(note_f1)) if note_f1 else 0.0,
        "exact_match": float(exact / len(predictions)) if len(predictions) else 0.0,
        "n_code_proposals": int(proposed),
        "mean_code_proposals_per_note": float(proposed / len(predictions)) if len(predictions) else 0.0,
    }


def select_threshold_max_recall_at_precision(
    validation_predictions: pd.DataFrame,
    *,
    target_precision: float = 0.95,
) -> float | None:
    """Select a validation-only proposal threshold.

    Primary objective: maximise micro recall subject to a prespecified micro precision.
    Ties prefer more code proposals and then the higher threshold. This policy concerns
    per-code suggestions, not automatic acceptance of an entire multi-label note.
    """
    if not 0 <= target_precision <= 1:
        raise ValueError("target_precision must be between 0 and 1")
    scores = sorted({
        float(item.get("score", 0.0) or 0.0)
        for value in validation_predictions.get("candidates_json", [])
        for item in _candidate_list(value)
    })
    feasible: list[tuple[float, int, float, float]] = []
    for threshold in scores:
        metrics = threshold_metrics(validation_predictions, threshold)
        if float(metrics["micro_precision"]) + 1e-12 >= target_precision:
            feasible.append((
                float(metrics["micro_recall"]),
                int(metrics["n_code_proposals"]),
                float(metrics["micro_precision"]),
                float(threshold),
            ))
    if not feasible:
        return None
    feasible.sort(reverse=True)
    return feasible[0][3]
