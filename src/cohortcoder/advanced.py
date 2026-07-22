from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .core import HistoricalCoder, Prediction


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(list(values.values()), dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return {key: 0.0 for key in values}
    return {key: (float(value) - lo) / (hi - lo) for key, value in values.items()}


class DenseSemanticIndex:
    """Optional sentence-transformer index for terminology and historical coding memory.

    The model name/path is supplied by the caller. Nothing is downloaded implicitly by
    the core package, and this backend is only constructed when explicitly requested.
    """

    def __init__(self, model_name: str, *, batch_size: int = 64, device: str | None = None) -> None:
        self.model_name = str(model_name)
        self.batch_size = int(batch_size)
        self.device = device
        self._model = None
        self.is_fitted = False

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Dense retrieval requires the optional 'sentence-transformers' dependency"
                ) from exc
            kwargs = {"device": self.device} if self.device else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    @staticmethod
    def _safe(values: Iterable[Any]) -> list[str]:
        return [str(value) if str(value).strip() else "__empty__" for value in values]

    def fit(self, history: pd.DataFrame, terminology: pd.DataFrame) -> "DenseSemanticIndex":
        model = self._load_model()
        self.terminology = terminology.drop_duplicates("code").reset_index(drop=True).copy()
        self.history = history.reset_index(drop=True).copy()
        term_column = "search_text" if "search_text" in self.terminology.columns else "term"
        history_text = (
            self.history["mention"].where(
                self.history["mention"].astype(str).str.strip().str.len() > 0,
                self.history["text"],
            )
            if "mention" in self.history.columns
            else self.history["text"]
        )
        self.term_embeddings = np.asarray(
            model.encode(
                self._safe(self.terminology[term_column]),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        self.history_embeddings = np.asarray(
            model.encode(
                self._safe(history_text),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        self.is_fitted = True
        return self

    def score(self, text: str) -> dict[str, dict[str, float]]:
        if not self.is_fitted:
            raise RuntimeError("DenseSemanticIndex must be fitted before scoring")
        model = self._load_model()
        query = np.asarray(
            model.encode(
                [str(text) if str(text).strip() else "__empty__"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )[0]
        term_scores = self.term_embeddings @ query
        history_scores = self.history_embeddings @ query
        history_by_code: dict[str, float] = {}
        for idx, score in enumerate(history_scores):
            code = str(self.history.iloc[idx]["gold_code"])
            history_by_code[code] = max(history_by_code.get(code, -1.0), float(score))
        out: dict[str, dict[str, float]] = {}
        for idx, row in self.terminology.iterrows():
            code = str(row["code"])
            out[code] = {
                "term_score": float(term_scores[idx]),
                "history_score": float(history_by_code.get(code, 0.0)),
            }
        return out


class CrossEncoderCandidateReranker:
    """Optional cross-encoder reranker over a frozen candidate set."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        self.model_name = str(model_name)
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Cross-encoder reranking requires the optional 'sentence-transformers' dependency"
                ) from exc
            kwargs = {"device": self.device} if self.device else {}
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def score_candidates(self, query: str, candidates: list[dict[str, Any]]) -> dict[str, float]:
        if not candidates:
            return {}
        pairs = [
            [str(query), str(item.get("search_text") or item.get("term") or "")]
            for item in candidates
        ]
        scores = self._load_model().predict(pairs)
        return {str(item["code"]): float(score) for item, score in zip(candidates, scores)}


@dataclass(frozen=True)
class AdvancedModelConfig:
    history_weight: float = 0.5
    dense_weight: float = 0.0
    reranker_weight: float = 0.0

    def complexity(self) -> tuple[int, int, float, float, float]:
        return (
            int(self.reranker_weight > 0),
            int(self.dense_weight > 0),
            self.reranker_weight,
            self.dense_weight,
            self.history_weight,
        )


class AdvancedSingleLabelCoder:
    """Fuse lexical, optional dense biomedical, and optional cross-encoder evidence.

    Candidate generation is separated from reranking. The code space is fixed by the
    supplied terminology table; neither dense retrieval nor the reranker can invent a
    code outside that dictionary.
    """

    def __init__(
        self,
        config: AdvancedModelConfig,
        *,
        top_k: int = 10,
        retrieval_pool: int = 50,
        dense_index: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.config = config
        self.top_k = int(top_k)
        self.retrieval_pool = max(int(retrieval_pool), self.top_k)
        self.dense_index = dense_index
        self.reranker = reranker

    def fit(self, history: pd.DataFrame, terminology: pd.DataFrame) -> "AdvancedSingleLabelCoder":
        self.history = history.reset_index(drop=True).copy()
        self.terminology = terminology.drop_duplicates("code").reset_index(drop=True).copy()
        self.lexical = HistoricalCoder(
            history_weight=float(self.config.history_weight),
            top_k=self.retrieval_pool,
        ).fit(self.history, self.terminology)
        if self.dense_index is not None and not getattr(self.dense_index, "is_fitted", False):
            self.dense_index.fit(self.history, self.terminology)
        return self

    def _candidate_text(self, code: str) -> str:
        row = self.terminology[self.terminology["code"].astype(str) == str(code)]
        if row.empty:
            return ""
        value = row.iloc[0]
        return str(value.get("search_text", value.get("term", "")))

    def _combined_scores(self, text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        lexical_prediction = self.lexical.predict_one(text)
        lexical_scores = {str(item["code"]): float(item["score"]) for item in lexical_prediction.candidates}
        dense_rows = self.dense_index.score(text) if self.dense_index is not None else {}
        dense_combined = {
            code: (1 - self.config.history_weight) * float(parts.get("term_score", 0.0))
            + self.config.history_weight * float(parts.get("history_score", 0.0))
            for code, parts in dense_rows.items()
        }
        dense_norm = _minmax(dense_combined)
        candidate_codes = set(lexical_scores)
        if dense_norm:
            candidate_codes.update(
                code for code, _ in sorted(dense_norm.items(), key=lambda item: item[1], reverse=True)[: self.retrieval_pool]
            )
        rows: list[dict[str, Any]] = []
        for code in candidate_codes:
            matched = self.terminology[self.terminology["code"].astype(str) == str(code)]
            if matched.empty:
                continue
            term = str(matched.iloc[0]["term"])
            lexical_score = float(lexical_scores.get(code, 0.0))
            dense_score = float(dense_norm.get(code, 0.0))
            combined = (1 - self.config.dense_weight) * lexical_score + self.config.dense_weight * dense_score
            rows.append({
                "code": code,
                "term": term,
                "search_text": self._candidate_text(code),
                "lexical_score": lexical_score,
                "dense_score": dense_score,
                "retrieval_score": float(combined),
            })
        rows.sort(key=lambda item: item["retrieval_score"], reverse=True)
        rows = rows[: self.retrieval_pool]
        if self.reranker is not None and self.config.reranker_weight > 0 and rows:
            raw = self.reranker.score_candidates(text, rows)
            rerank_norm = _minmax(raw)
            for item in rows:
                item["reranker_score"] = float(rerank_norm.get(str(item["code"]), 0.0))
                item["score"] = (
                    (1 - self.config.reranker_weight) * float(item["retrieval_score"])
                    + self.config.reranker_weight * float(item["reranker_score"])
                )
        else:
            for item in rows:
                item["reranker_score"] = 0.0
                item["score"] = float(item["retrieval_score"])
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows, lexical_prediction.historical_cases

    def predict_one(self, text: str) -> Prediction:
        rows, historical_cases = self._combined_scores(str(text))
        if not rows:
            raise RuntimeError("No candidates available")
        candidates = rows[: self.top_k]
        best = candidates[0]
        second = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = max(0.0, float(best["score"]) - float(second))
        confidence = float(np.clip(0.5 * float(best["score"]) + 0.5 * margin, 0.0, 1.0))
        return Prediction(
            str(best["code"]),
            str(best["term"]),
            confidence,
            candidates,
            historical_cases,
        )

    def predict(self, texts: Iterable[str]) -> list[Prediction]:
        return [self.predict_one(str(text)) for text in texts]

    def score_code(self, text: str, code: str) -> float:
        rows, _ = self._combined_scores(str(text))
        for item in rows:
            if str(item["code"]) == str(code):
                return float(item["score"])
        return 0.0
