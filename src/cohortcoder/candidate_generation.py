from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .core import Prediction


def _nonempty(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "__empty__"


def _split_aliases(row: pd.Series) -> list[str]:
    values: list[str] = []
    term = str(row.get("term", "") or "").strip()
    if term:
        values.append(term)
    synonyms = str(row.get("synonyms", "") or "")
    values.extend(part.strip() for part in synonyms.split("|") if part.strip())
    # Preserve order while removing case-insensitive duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique or ["__empty__"]


@dataclass(frozen=True)
class CandidateGenerationConfig:
    history_weight: float = 0.5
    word_weight: float = 0.2
    top_k: int = 50

    def validate(self) -> None:
        if not 0 <= self.history_weight <= 1:
            raise ValueError("history_weight must be between 0 and 1")
        if not 0 <= self.word_weight <= 1:
            raise ValueError("word_weight must be between 0 and 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


class AliasAwareHybridCoder:
    """Candidate generator that retrieves terminology aliases before aggregating to codes.

    v0.1.1 stored many TRAIN-derived aliases in ``synonyms`` but the baseline terminology
    vectorizer indexed only one ``term`` string per code. This model indexes every alias as
    its own retrieval unit, combines character and word TF-IDF similarity at alias level,
    then takes the best alias score for each code before blending historical coding memory.

    The model remains deterministic, TRAIN-only for candidate construction, and auditable:
    each candidate records the matched alias plus char/word/history component scores.
    """

    def __init__(
        self,
        *,
        history_weight: float = 0.5,
        word_weight: float = 0.2,
        top_k: int = 50,
    ) -> None:
        self.config = CandidateGenerationConfig(
            history_weight=float(history_weight),
            word_weight=float(word_weight),
            top_k=int(top_k),
        )
        self.config.validate()

    @property
    def history_weight(self) -> float:
        return self.config.history_weight

    @property
    def word_weight(self) -> float:
        return self.config.word_weight

    @property
    def top_k(self) -> int:
        return self.config.top_k

    def fit(self, history: pd.DataFrame, terminology: pd.DataFrame) -> "AliasAwareHybridCoder":
        required_history = {"text", "gold_code", "gold_term"}
        required_terms = {"code", "term"}
        if not required_history.issubset(history.columns):
            raise ValueError(f"history requires columns: {sorted(required_history)}")
        if not required_terms.issubset(terminology.columns):
            raise ValueError(f"terminology requires columns: {sorted(required_terms)}")
        if history.empty:
            raise ValueError("history must contain at least one record")
        if terminology.empty:
            raise ValueError("terminology must contain at least one code")

        self.history = history.reset_index(drop=True).copy()
        self.terminology = terminology.drop_duplicates("code").reset_index(drop=True).copy()

        aliases: list[str] = []
        alias_codes: list[str] = []
        for _, row in self.terminology.iterrows():
            code = str(row["code"])
            for alias in _split_aliases(row):
                aliases.append(_nonempty(alias))
                alias_codes.append(code)
        self.aliases = aliases
        self.alias_codes = np.asarray(alias_codes, dtype=object)

        self.alias_char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
        )
        self.alias_word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        self.alias_char_matrix = self.alias_char_vectorizer.fit_transform(self.aliases)
        self.alias_word_matrix = self.alias_word_vectorizer.fit_transform(self.aliases)

        if "mention" in self.history.columns:
            history_values = self.history["mention"].where(
                self.history["mention"].fillna("").astype(str).str.strip().str.len() > 0,
                self.history["text"],
            )
        else:
            history_values = self.history["text"]
        history_texts = [_nonempty(value) for value in history_values]
        self.history_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
        )
        self.history_matrix = self.history_vectorizer.fit_transform(history_texts)
        return self

    def _alias_scores(self, text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = _nonempty(text)
        char_scores = cosine_similarity(
            self.alias_char_vectorizer.transform([query]), self.alias_char_matrix
        )[0]
        word_scores = cosine_similarity(
            self.alias_word_vectorizer.transform([query]), self.alias_word_matrix
        )[0]
        fused = (1 - self.word_weight) * char_scores + self.word_weight * word_scores
        return fused, char_scores, word_scores

    def _history_scores(self, text: str) -> np.ndarray:
        query = _nonempty(text)
        return cosine_similarity(
            self.history_vectorizer.transform([query]), self.history_matrix
        )[0]

    def _best_alias_by_code(
        self,
        fused: np.ndarray,
        char_scores: np.ndarray,
        word_scores: np.ndarray,
    ) -> dict[str, dict]:
        best: dict[str, dict] = {}
        for idx, score in enumerate(fused):
            code = str(self.alias_codes[idx])
            current = best.get(code)
            if current is None or float(score) > float(current["terminology_score"]):
                best[code] = {
                    "terminology_score": float(score),
                    "char_score": float(char_scores[idx]),
                    "word_score": float(word_scores[idx]),
                    "matched_alias": str(self.aliases[idx]),
                }
        return best

    def _history_by_code(self, scores: np.ndarray) -> dict[str, float]:
        result: dict[str, float] = defaultdict(float)
        for idx, score in enumerate(scores):
            code = str(self.history.iloc[idx]["gold_code"])
            result[code] = max(float(result[code]), float(score))
        return dict(result)

    def score_code(self, text: str, code: str) -> float:
        fused, char_scores, word_scores = self._alias_scores(text)
        alias_by_code = self._best_alias_by_code(fused, char_scores, word_scores)
        history_by_code = self._history_by_code(self._history_scores(text))
        term = float(alias_by_code.get(str(code), {}).get("terminology_score", 0.0))
        hist = float(history_by_code.get(str(code), 0.0))
        return float((1 - self.history_weight) * term + self.history_weight * hist)

    def predict_one(self, text: str) -> Prediction:
        fused, char_scores, word_scores = self._alias_scores(text)
        alias_by_code = self._best_alias_by_code(fused, char_scores, word_scores)
        hist_scores = self._history_scores(text)
        history_by_code = self._history_by_code(hist_scores)

        rows: list[dict] = []
        for _, row in self.terminology.iterrows():
            code = str(row["code"])
            alias_info = alias_by_code.get(code, {})
            terminology_score = float(alias_info.get("terminology_score", 0.0))
            historical_score = float(history_by_code.get(code, 0.0))
            final_score = (
                (1 - self.history_weight) * terminology_score
                + self.history_weight * historical_score
            )
            rows.append(
                {
                    "code": code,
                    "term": str(row["term"]),
                    "system": str(row.get("system", "")),
                    "score": float(final_score),
                    "terminology_score": terminology_score,
                    "history_score": historical_score,
                    "char_score": float(alias_info.get("char_score", 0.0)),
                    "word_score": float(alias_info.get("word_score", 0.0)),
                    "matched_alias": str(alias_info.get("matched_alias", "")),
                }
            )
        rows.sort(key=lambda item: (item["score"], item["terminology_score"]), reverse=True)
        candidates = rows[: self.top_k]
        best = candidates[0]
        second = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        margin = max(0.0, float(best["score"]) - second)
        confidence = float(np.clip(0.5 * float(best["score"]) + 0.5 * margin, 0, 1))

        hist_idx = np.argsort(-hist_scores)[: min(5, len(hist_scores))]
        historical_cases = [
            {
                "text": str(self.history.iloc[i]["text"]),
                "mention": str(self.history.iloc[i].get("mention", "")),
                "code": str(self.history.iloc[i]["gold_code"]),
                "term": str(self.history.iloc[i]["gold_term"]),
                "similarity": float(hist_scores[i]),
            }
            for i in hist_idx
        ]
        return Prediction(
            str(best["code"]),
            str(best["term"]),
            confidence,
            candidates,
            historical_cases,
        )

    def predict(self, texts: Iterable[str]) -> list[Prediction]:
        return [self.predict_one(str(text)) for text in texts]
