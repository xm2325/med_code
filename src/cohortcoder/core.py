from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Prediction:
    code: str
    term: str
    confidence: float
    candidates: list[dict]
    historical_cases: list[dict]


class HistoricalCoder:
    """Auditable retrieval baseline combining terminology and historical coding memory.

    This is intentionally a transparent baseline. More complex dense encoders,
    cross-encoders, or LLM rerankers should be compared against the same held-out
    split rather than assumed to improve performance.
    """

    def __init__(self, history_weight: float = 0.5, top_k: int = 10):
        if not 0 <= history_weight <= 1:
            raise ValueError("history_weight must be between 0 and 1")
        self.history_weight = history_weight
        self.top_k = top_k

    def fit(self, history: pd.DataFrame, terminology: pd.DataFrame) -> "HistoricalCoder":
        required_history = {"text", "gold_code", "gold_term"}
        required_terms = {"code", "term"}
        if not required_history.issubset(history.columns):
            raise ValueError(f"history requires columns: {sorted(required_history)}")
        if not required_terms.issubset(terminology.columns):
            raise ValueError(f"terminology requires columns: {sorted(required_terms)}")

        self.history = history.reset_index(drop=True).copy()
        self.terminology = terminology.drop_duplicates("code").reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        corpus = self.terminology["term"].fillna("").tolist() + self.history["text"].fillna("").tolist()
        self.vectorizer.fit(corpus)
        self.term_matrix = self.vectorizer.transform(self.terminology["term"].fillna(""))
        self.history_matrix = self.vectorizer.transform(self.history["text"].fillna(""))
        return self

    def predict_one(self, text: str) -> Prediction:
        query = self.vectorizer.transform([text])
        term_scores = cosine_similarity(query, self.term_matrix)[0]
        hist_scores = cosine_similarity(query, self.history_matrix)[0]

        history_by_code: dict[str, float] = {}
        for idx, score in enumerate(hist_scores):
            code = str(self.history.iloc[idx]["gold_code"])
            history_by_code[code] = max(history_by_code.get(code, 0.0), float(score))

        rows = []
        for idx, row in self.terminology.iterrows():
            code = str(row["code"])
            terminology_score = float(term_scores[idx])
            historical_score = history_by_code.get(code, 0.0)
            final_score = (1 - self.history_weight) * terminology_score + self.history_weight * historical_score
            rows.append({
                "code": code,
                "term": str(row["term"]),
                "score": final_score,
                "terminology_score": terminology_score,
                "history_score": historical_score,
            })
        rows.sort(key=lambda x: x["score"], reverse=True)
        candidates = rows[: self.top_k]
        best = candidates[0]
        second = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = max(0.0, best["score"] - second)
        confidence = float(np.clip(0.5 * best["score"] + 0.5 * margin, 0, 1))

        hist_idx = np.argsort(-hist_scores)[: min(5, len(hist_scores))]
        historical_cases = [
            {
                "text": str(self.history.iloc[i]["text"]),
                "code": str(self.history.iloc[i]["gold_code"]),
                "term": str(self.history.iloc[i]["gold_term"]),
                "similarity": float(hist_scores[i]),
            }
            for i in hist_idx
        ]
        return Prediction(best["code"], best["term"], confidence, candidates, historical_cases)

    def predict(self, texts: Iterable[str]) -> list[Prediction]:
        return [self.predict_one(str(text)) for text in texts]


def accuracy_at_k(gold: Iterable[str], candidate_lists: Iterable[list[dict]], k: int = 1) -> float:
    gold = list(map(str, gold))
    candidate_lists = list(candidate_lists)
    if not gold:
        return float("nan")
    hits = [g in [str(c["code"]) for c in candidates[:k]] for g, candidates in zip(gold, candidate_lists)]
    return float(np.mean(hits))


def coverage_accuracy(gold: Iterable[str], predictions: Iterable[Prediction], threshold: float) -> dict:
    gold = list(map(str, gold))
    predictions = list(predictions)
    accepted = [i for i, p in enumerate(predictions) if p.confidence >= threshold]
    if not accepted:
        return {"coverage": 0.0, "accuracy": float("nan"), "n_auto": 0}
    correct = [predictions[i].code == gold[i] for i in accepted]
    return {
        "coverage": len(accepted) / len(predictions),
        "accuracy": float(np.mean(correct)),
        "n_auto": len(accepted),
    }
