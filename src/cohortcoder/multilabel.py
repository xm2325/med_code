from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
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

    Terminology and historical text use separate vector spaces. To avoid comparing every
    new note with every TRAIN note during ranking, historical expert coding is aggregated
    into one sparse text centroid per code. Raw TRAIN notes are retained only for on-demand
    explanation provenance retrieval.
    """

    def __init__(self, history_weight: float = 0.5, top_k: int = 100, max_history_features: int = 100_000):
        if not 0 <= history_weight <= 1:
            raise ValueError("history_weight must be between 0 and 1")
        self.history_weight = float(history_weight)
        self.top_k = int(top_k)
        self.max_history_features = int(max_history_features)

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
        # Word n-grams are substantially more scalable than unrestricted character
        # n-grams for hundreds of thousands of long discharge summaries.
        self.history_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1 if len(history_texts) < 50 else 2,
            max_features=self.max_history_features,
            sublinear_tf=True,
        )
        self.term_matrix = self.term_vectorizer.fit_transform(term_texts)
        self.history_matrix = self.history_vectorizer.fit_transform(history_texts)
        self.history_codes = [parse_code_list(value) for value in self.history["gold_codes_json"]]
        self.code_to_term = dict(zip(self.terminology["code"].astype(str), self.terminology["term"].astype(str)))
        self.code_to_index = {str(code): idx for idx, code in enumerate(self.terminology["code"].astype(str))}

        counts: dict[str, int] = {}
        for codes in self.history_codes:
            for code in set(codes):
                if code in self.code_to_index:
                    counts[code] = counts.get(code, 0) + 1
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for doc_idx, codes in enumerate(self.history_codes):
            for code in set(codes):
                code_idx = self.code_to_index.get(code)
                if code_idx is None:
                    continue
                rows.append(code_idx)
                cols.append(doc_idx)
                values.append(1.0 / counts[code])
        incidence = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.terminology), len(self.history)),
            dtype=float,
        )
        self.code_history_matrix = incidence @ self.history_matrix
        return self

    def _score_vectors(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        query_text = str(text or "").strip() or "__empty__"
        term_scores = cosine_similarity(self.term_vectorizer.transform([query_text]), self.term_matrix)[0]
        history_scores = cosine_similarity(
            self.history_vectorizer.transform([query_text]),
            self.code_history_matrix,
        )[0]
        return term_scores, history_scores

    def score_code(self, text: str, code: str) -> float:
        idx = self.code_to_index.get(str(code))
        if idx is None:
            return 0.0
        term_scores, history_scores = self._score_vectors(text)
        return float((1.0 - self.history_weight) * term_scores[idx] + self.history_weight * history_scores[idx])

    def rank_one(self, text: str) -> MultiLabelPrediction:
        term_scores, history_scores = self._score_vectors(text)
        candidates: list[dict] = []
        for idx, row in self.terminology.iterrows():
            terminology_score = float(term_scores[idx])
            history_score = float(history_scores[idx])
            score = (1.0 - self.history_weight) * terminology_score + self.history_weight * history_score
            candidates.append({
                "code": str(row["code"]),
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
    """Select a validation-only per-code proposal threshold."""
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
