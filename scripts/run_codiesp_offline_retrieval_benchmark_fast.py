#!/usr/bin/env python3
"""Efficient CPU-only CodiEsp DEV retrieval benchmark; no LLM/API calls.

This is the computationally tractable two-stage version of the v0.2 benchmark.
It preserves the same four ablation questions while avoiding dense
fragment-by-all-ICD similarity matrices:

1. whole_note_lexical
2. sentence_hybrid
3. fragment_hybrid_alias
4. fragment_hybrid_alias_hierarchy

For hybrid methods, a high-recall sparse candidate pool is formed from whole-note
BM25, whole-note character TF-IDF and max-over-local BM25. Character and latent
semantic local reranking are then computed only inside that pool.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from run_codiesp_icd_kb_ab import build_kb
from run_codiesp_offline_retrieval_benchmark import (
    BM25Sparse,
    TOP_KS,
    build_augmented_docs,
    deterministic_fragments,
    evaluate_method,
    expand_abbreviations,
    load_codiesp_split,
    norm_code,
    normalise_space,
    split_sentences,
)

SCHEMA_VERSION = "codiesp-offline-retrieval-fast-v0.3"
METHODS = [
    "whole_note_lexical",
    "sentence_hybrid",
    "fragment_hybrid_alias",
    "fragment_hybrid_alias_hierarchy",
]


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).any():
        return np.zeros_like(x)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi <= lo + 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def sparse_row_max(matrix) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    mx = matrix.max(axis=1)
    if sparse.issparse(mx):
        return mx.toarray().ravel().astype(np.float32)
    return np.asarray(mx).ravel().astype(np.float32)


def top_indices(scores: np.ndarray, n: int) -> np.ndarray:
    n = min(int(n), len(scores))
    if n <= 0:
        return np.array([], dtype=int)
    if n == len(scores):
        return np.argsort(-scores)
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx])]


class EfficientTwoStageRetriever:
    def __init__(self, kb: list[dict], *, lsa_dim: int = 64, candidate_n: int = 1500):
        self.kb = kb
        self.candidate_n = candidate_n
        self.docs = build_augmented_docs(kb)
        self.bm25 = BM25Sparse(self.docs)

        self.char = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=160_000,
            dtype=np.float32,
        )
        self.x_char = self.char.fit_transform(self.docs).tocsr()

        self.word = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=120_000,
            dtype=np.float32,
        )
        self.x_word = self.word.fit_transform(self.docs).tocsr()
        max_dim = min(int(lsa_dim), max(2, self.x_word.shape[0] - 1), max(2, self.x_word.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=max_dim, n_iter=4, random_state=17)
        self.x_lsa = normalize(self.svd.fit_transform(self.x_word)).astype(np.float32)

        self.code_to_idx = {r["code"]: i for i, r in enumerate(kb)}
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, set[str]] = defaultdict(set)
        for row in kb:
            code = row["code"]
            for parent in row.get("hierarchy", []):
                pc = norm_code(parent.get("code", ""))
                if pc in self.code_to_idx:
                    self.parents[code].add(pc)
                    self.children[pc].add(code)

    def _char_full(self, query: str) -> np.ndarray:
        q = self.char.transform([query])
        return np.asarray((self.x_char @ q.T).toarray()).ravel().astype(np.float32)

    def _local_char(self, queries: list[str], candidates: np.ndarray) -> np.ndarray:
        if not queries or len(candidates) == 0:
            return np.zeros(len(candidates), dtype=np.float32)
        q = self.char.transform(queries)
        return sparse_row_max(self.x_char[candidates] @ q.T)

    def _local_lsa(self, queries: list[str], candidates: np.ndarray) -> np.ndarray:
        if not queries or len(candidates) == 0:
            return np.zeros(len(candidates), dtype=np.float32)
        q = normalize(self.svd.transform(self.word.transform(queries))).astype(np.float32)
        return np.max(self.x_lsa[candidates] @ q.T, axis=1).astype(np.float32)

    def _candidate_union(self, *channels: np.ndarray) -> np.ndarray:
        chosen: set[int] = set()
        for scores in channels:
            chosen.update(int(x) for x in top_indices(scores, self.candidate_n))
        return np.array(sorted(chosen), dtype=int)

    def _hierarchy_expand(self, scores: np.ndarray, seed_n: int = 40) -> np.ndarray:
        out = scores.copy()
        for idx in top_indices(scores, seed_n):
            code = self.kb[int(idx)]["code"]
            seed_score = float(scores[int(idx)])
            for pc in self.parents.get(code, set()):
                pidx = self.code_to_idx[pc]
                out[pidx] = max(out[pidx], seed_score * 0.92)
            for cc in self.children.get(code, set()):
                cidx = self.code_to_idx[cc]
                out[cidx] = max(out[cidx], seed_score * 0.84)
        return out

    def retrieve(self, text: str, method: str, top_k: int = 35) -> list[dict]:
        whole = normalise_space(text)
        whole_bm25_raw = self.bm25.score(whole)
        whole_char_raw = self._char_full(whole)
        whole_bm25 = minmax(whole_bm25_raw)
        whole_char = minmax(whole_char_raw)

        if method == "whole_note_lexical":
            score = 0.65 * whole_bm25 + 0.35 * whole_char
        else:
            if method == "sentence_hybrid":
                local_queries = split_sentences(text)[:80]
            elif method in {"fragment_hybrid_alias", "fragment_hybrid_alias_hierarchy"}:
                local_queries = [expand_abbreviations(x) for x in deterministic_fragments(text, max_fragments=80)]
            else:
                raise ValueError(f"Unknown method: {method}")

            local_bm25_raw = self.bm25.max_score(local_queries)
            candidates = self._candidate_union(whole_bm25_raw, whole_char_raw, local_bm25_raw)
            local_char = minmax(self._local_char(local_queries, candidates))
            local_lsa = minmax(self._local_lsa(local_queries, candidates))
            local_bm25 = minmax(local_bm25_raw[candidates])

            score = np.zeros(len(self.kb), dtype=np.float32)
            score[candidates] = (
                0.10 * whole_bm25[candidates]
                + 0.10 * whole_char[candidates]
                + 0.40 * local_bm25
                + 0.20 * local_char
                + 0.20 * local_lsa
            )
            if method == "fragment_hybrid_alias_hierarchy":
                score = self._hierarchy_expand(score)

        order = np.argsort(-score)[:top_k]
        return [
            {
                "rank": rank,
                "code": self.kb[int(idx)]["code"],
                "description": self.kb[int(idx)]["description"],
                "score": float(score[int(idx)]),
                "hierarchy": self.kb[int(idx)].get("hierarchy", [])[-4:],
                "coding_notes": self.kb[int(idx)].get("coding_notes", [])[:8],
            }
            for rank, idx in enumerate(order, start=1)
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-test-eval", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/codiesp_offline_retrieval/dev_v03_fast"))
    parser.add_argument("--lsa-dim", type=int, default=64)
    parser.add_argument("--candidate-n", type=int, default=1500)
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test_eval:
        raise SystemExit("TEST evaluation is blocked. Tune on DEV only.")

    print("CPU/OFFLINE ONLY: no DeepSeek, no LLM, no API inference.")
    texts, gold, corpus_manifest = load_codiesp_split(args.split)
    if args.limit > 0:
        ids = sorted(texts)[: args.limit]
        texts = {k: texts[k] for k in ids}
        gold = {k: gold[k] for k in ids}

    kb, kb_manifest = build_kb()
    print(f"Cases={len(texts)}; ICD KB={len(kb)}; candidate_n={args.candidate_n}")
    retriever = EfficientTwoStageRetriever(kb, lsa_dim=args.lsa_dim, candidate_n=args.candidate_n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    case_frames = []
    miss_frames = []
    for method in METHODS:
        summary, per_case, misses = evaluate_method(texts, gold, retriever, method, max(TOP_KS))
        summaries.append(summary)
        cdf = pd.DataFrame(per_case)
        cdf.insert(0, "method", method)
        case_frames.append(cdf)
        mdf = pd.DataFrame(misses)
        if not mdf.empty:
            mdf.insert(0, "method", method)
            miss_frames.append(mdf)
        # checkpoint after each method
        pd.DataFrame(summaries).to_csv(args.output_dir / "retrieval_summary_partial.csv", index=False)
        print(json.dumps(summary, indent=2))

    summary_df = pd.DataFrame(summaries).sort_values(
        ["micro_recall_at_20", "micro_recall_at_35"], ascending=False
    )
    summary_df.to_csv(args.output_dir / "retrieval_summary.csv", index=False)
    pd.concat(case_frames, ignore_index=True).to_csv(args.output_dir / "per_case_retrieval.csv", index=False)
    if miss_frames:
        misses_df = pd.concat(miss_frames, ignore_index=True)
        misses_df.to_csv(args.output_dir / "retrieval_misses.csv", index=False)
        misses_df.groupby(["method", "error_category"]).size().reset_index(name="n").to_csv(
            args.output_dir / "retrieval_miss_categories.csv", index=False
        )

    best = summary_df.iloc[0].to_dict()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": args.split,
        "n_cases": len(texts),
        "candidate_n": args.candidate_n,
        "lsa_dim": args.lsa_dim,
        "methods": METHODS,
        "selection_rule": "DEV-only: maximise micro Recall@20, tie-break by micro Recall@35",
        "selected_method": best["method"],
        "selected_metrics": best,
        "corpus": corpus_manifest,
        "icd_kb": kb_manifest,
        "no_llm_api_calls": True,
        "deepseek_api_key_read": False,
        "hybrid_retrieval_note": "local char/LSA reranking is restricted to a union candidate pool from sparse high-recall channels",
    }
    (args.output_dir / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("\n=== DEV RETRIEVAL SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"Selected method: {best['method']}")
    print("No DeepSeek/LLM/API calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
