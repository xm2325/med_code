#!/usr/bin/env python3
"""Offline CodiEsp ICD retrieval benchmark: no LLM/API calls.

Purpose
-------
Improve the *retrieval* stage before spending any DeepSeek/API budget.  The
benchmark uses CodiEsp DEV by default and compares progressively stronger
retrieval pipelines inspired by multi-stage clinical coding literature:

1. whole_note_lexical
   Whole-note BM25 + character TF-IDF baseline.
2. sentence_hybrid
   Whole-note + sentence-level BM25/character/latent-semantic retrieval.
3. fragment_hybrid_alias
   Deterministic evidence-fragment decomposition + query alias expansion.
4. fragment_hybrid_alias_hierarchy
   As above, with ICD hierarchy neighbourhood expansion after retrieval.

This script intentionally does NOT import or call any DeepSeek client, does
not read DEEPSEEK_API_KEY, and does not perform LLM inference.

Scientific guardrail
--------------------
DEV is the default split for method development. TEST evaluation is blocked
unless --allow-test-eval is supplied explicitly. Do not tune retrieval using
TEST gold labels.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from run_codiesp_icd_kb_ab import (
    CODIESP_CORPUS_URL,
    build_kb,
    fetch_bytes,
    norm_code,
    sha256,
)

SCHEMA_VERSION = "codiesp-offline-retrieval-v0.2"
TOP_KS = (5, 10, 20, 35)

# Small deterministic query-side alias dictionary. These aliases do not use
# test labels and are only query expansion. KB-side aliases additionally come
# from official/CodiEsp descriptions, inclusion terms, notes, and hierarchy.
ABBREVIATION_EXPANSIONS = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "copd": "chronic obstructive pulmonary disease",
    "chf": "congestive heart failure",
    "hf": "heart failure",
    "ckd": "chronic kidney disease",
    "uti": "urinary tract infection",
    "mi": "myocardial infarction",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "af": "atrial fibrillation",
    "afib": "atrial fibrillation",
    "bph": "benign prostatic hyperplasia",
    "ra": "rheumatoid arthritis",
    "oa": "osteoarthritis",
    "gerd": "gastroesophageal reflux disease",
}

CUE_PATTERNS = [
    r"(?:diagnosis|diagnosed|history|antecedent|background)\s+(?:of|with)\s+([^.;:\n]{3,120})",
    r"(?:compatible with|consistent with|revealed|showed|demonstrated|confirmed)\s+([^.;:\n]{3,120})",
    r"(?:pathology|biopsy|cytology|histology)\s*(?:showed|confirmed|revealed|:)\s*([^.;:\n]{3,120})",
]


def load_codiesp_split(split: str) -> tuple[dict[str, str], dict[str, set[str]], dict]:
    """Load English CodiEsp text and diagnosis gold for dev/test."""
    split = split.lower().strip()
    if split not in {"dev", "test", "train"}:
        raise ValueError(f"Unsupported split: {split}")
    raw = fetch_bytes(CODIESP_CORPUS_URL)
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = z.namelist()

    texts: dict[str, str] = {}
    needle = f"/{split}/text_files_en/"
    for name in names:
        low = "/" + name.lower().lstrip("/")
        if needle in low and low.endswith(".txt"):
            texts[Path(name).stem] = z.read(name).decode("utf-8", errors="replace")

    label_name = f"{split}d.tsv".lower()
    candidates = [n for n in names if Path(n).name.lower() == label_name]
    if not candidates:
        raise RuntimeError(f"Missing {split}D.tsv in CodiEsp archive")
    label_path = sorted(candidates, key=len)[0]
    label_raw = z.read(label_path).decode("utf-8", errors="replace")
    gold: dict[str, set[str]] = defaultdict(set)
    for line in label_raw.splitlines():
        p = line.strip().split("\t")
        if len(p) >= 2:
            gold[p[0]].add(norm_code(p[1]))

    common_ids = sorted(set(texts) & set(gold))
    texts = {k: texts[k] for k in common_ids}
    gold = {k: set(gold[k]) for k in common_ids}
    manifest = {
        "url": CODIESP_CORPUS_URL,
        "sha256": sha256(raw),
        "split": split,
        "n_texts": len(texts),
        "n_gold_codes": sum(len(v) for v in gold.values()),
        "label_file": label_path,
    }
    return texts, gold, manifest


def normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def expand_abbreviations(text: str) -> str:
    """Append deterministic expansions for abbreviations present in text."""
    low = f" {text.lower()} "
    expansions: list[str] = []
    for abbr, expanded in ABBREVIATION_EXPANSIONS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(abbr)}(?![a-z0-9])", low):
            expansions.append(expanded)
    return normalise_space(text + (" | aliases: " + " ; ".join(expansions) if expansions else ""))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text)
    return [normalise_space(x) for x in parts if 3 <= len(normalise_space(x))]


def deterministic_fragments(text: str, max_fragments: int = 100) -> list[str]:
    """Create high-recall evidence-like fragments without an NLP/LLM model.

    We use sentences, semicolon/colon clauses, selected comma clauses, and
    cue-triggered spans. This is intentionally called *fragment decomposition*,
    not clinical NER, because it is deterministic and unsupervised.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        x = normalise_space(x).strip(" -,:;")
        if not x or len(x) < 4 or len(x.split()) > 45:
            return
        key = x.lower()
        if key not in seen:
            seen.add(key)
            out.append(x)

    for sent in split_sentences(text):
        add(sent)
        for clause in re.split(r"\s*[;:]\s*", sent):
            add(clause)
        # Comma splitting is limited to avoid exploding narrative prose.
        comma_parts = [normalise_space(x) for x in sent.split(",")]
        if 2 <= len(comma_parts) <= 6:
            for part in comma_parts:
                if 2 <= len(part.split()) <= 25:
                    add(part)

    low = normalise_space(text)
    for pattern in CUE_PATTERNS:
        for m in re.finditer(pattern, low, flags=re.IGNORECASE):
            add(m.group(1))

    return out[:max_fragments]


def build_augmented_docs(kb: list[dict]) -> list[str]:
    """Build KB documents with descriptions, aliases, inclusion terms and hierarchy."""
    docs: list[str] = []
    for row in kb:
        pieces = [
            row.get("code", ""),
            row.get("description", ""),
            row.get("codiesp_en_description", ""),
        ]
        # Notes often include InclusionTerm/Includes/Excludes/CodeFirst text.
        pieces.extend(row.get("coding_notes", []))
        for p in row.get("hierarchy", []):
            pieces.append(p.get("description", ""))
        text = " | ".join(x for x in pieces if x)
        # Lightweight description-derived alias normalisation.
        text += " | " + re.sub(r"[,()/\-]", " ", text)
        docs.append(normalise_space(text))
    return docs


class BM25Sparse:
    """Sparse BM25 document index with batched max-over-query scoring."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
            ngram_range=(1, 2),
            min_df=1,
            max_features=180_000,
        )
        counts = self.vectorizer.fit_transform(docs).tocsr().astype(np.float32)
        n_docs = counts.shape[0]
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        avgdl = float(doc_len.mean()) if n_docs else 1.0
        binary = counts.copy()
        binary.data[:] = 1.0
        df = np.asarray(binary.sum(axis=0)).ravel()
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        coo = counts.tocoo(copy=True)
        denom = coo.data + k1 * (1.0 - b + b * doc_len[coo.row] / max(avgdl, 1e-6))
        coo.data = (coo.data * (k1 + 1.0) / np.maximum(denom, 1e-6)) * idf[coo.col]
        self.matrix = coo.tocsr()

    def score(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query]).tocsr().astype(np.float32)
        if q.nnz:
            q.data[:] = 1.0
        return np.asarray((self.matrix @ q.T).toarray()).ravel()

    def max_score(self, queries: list[str]) -> np.ndarray:
        if not queries:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        q = self.vectorizer.transform(queries).tocsr().astype(np.float32)
        if q.nnz:
            q.data[:] = 1.0
        sims = self.matrix @ q.T
        # scipy sparse max returns a sparse/matrix object depending on version.
        return np.asarray(sims.max(axis=1).toarray()).ravel()


class OfflineHybridRetriever:
    def __init__(self, kb: list[dict], lsa_dim: int = 128):
        self.kb = kb
        self.docs = build_augmented_docs(kb)
        self.bm25 = BM25Sparse(self.docs)
        self.char = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=180_000,
        )
        self.x_char = self.char.fit_transform(self.docs)
        self.word = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=140_000,
        )
        self.x_word = self.word.fit_transform(self.docs)
        max_dim = min(lsa_dim, max(2, self.x_word.shape[0] - 1), max(2, self.x_word.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=max_dim, random_state=17)
        self.x_lsa = normalize(self.svd.fit_transform(self.x_word))

        self.code_to_idx = {r["code"]: i for i, r in enumerate(kb)}
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, set[str]] = defaultdict(set)
        for r in kb:
            code = r["code"]
            for p in r.get("hierarchy", []):
                pc = norm_code(p.get("code", ""))
                if pc in self.code_to_idx:
                    self.parents[code].add(pc)
                    self.children[pc].add(code)

    def _char_score(self, query: str) -> np.ndarray:
        return cosine_similarity(self.char.transform([query]), self.x_char)[0]

    def _char_max(self, queries: list[str]) -> np.ndarray:
        if not queries:
            return np.zeros(len(self.kb))
        return cosine_similarity(self.char.transform(queries), self.x_char).max(axis=0)

    def _lsa_score(self, query: str) -> np.ndarray:
        q = normalize(self.svd.transform(self.word.transform([query])))
        return (self.x_lsa @ q.T).ravel()

    def _lsa_max(self, queries: list[str]) -> np.ndarray:
        if not queries:
            return np.zeros(len(self.kb))
        q = normalize(self.svd.transform(self.word.transform(queries)))
        return (self.x_lsa @ q.T).max(axis=1)

    @staticmethod
    def _rank_normalise(x: np.ndarray) -> np.ndarray:
        """Robust 0-1 scaling so heterogeneous channels can be combined."""
        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).any():
            return np.zeros_like(x)
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if hi <= lo + 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def _hierarchy_expand(self, scores: np.ndarray, seed_n: int = 30) -> np.ndarray:
        out = scores.copy()
        seeds = np.argsort(-scores)[:seed_n]
        for idx in seeds:
            code = self.kb[int(idx)]["code"]
            s = float(scores[int(idx)])
            for pc in self.parents.get(code, set()):
                pidx = self.code_to_idx[pc]
                out[pidx] = max(out[pidx], s * 0.92)
            for cc in self.children.get(code, set()):
                cidx = self.code_to_idx[cc]
                out[cidx] = max(out[cidx], s * 0.84)
        return out

    def retrieve(self, text: str, method: str, top_k: int = 35) -> list[dict]:
        whole = normalise_space(text)
        sentences = split_sentences(text)[:100]
        fragments = deterministic_fragments(text)[:100]
        alias_fragments = [expand_abbreviations(x) for x in fragments]

        whole_bm25 = self._rank_normalise(self.bm25.score(whole))
        whole_char = self._rank_normalise(self._char_score(whole))

        if method == "whole_note_lexical":
            score = 0.65 * whole_bm25 + 0.35 * whole_char

        elif method == "sentence_hybrid":
            sent_bm25 = self._rank_normalise(self.bm25.max_score(sentences))
            sent_char = self._rank_normalise(self._char_max(sentences))
            sent_lsa = self._rank_normalise(self._lsa_max(sentences))
            score = 0.10 * whole_bm25 + 0.10 * whole_char + 0.40 * sent_bm25 + 0.20 * sent_char + 0.20 * sent_lsa

        elif method in {"fragment_hybrid_alias", "fragment_hybrid_alias_hierarchy"}:
            frag_bm25 = self._rank_normalise(self.bm25.max_score(alias_fragments))
            frag_char = self._rank_normalise(self._char_max(alias_fragments))
            frag_lsa = self._rank_normalise(self._lsa_max(alias_fragments))
            score = 0.08 * whole_bm25 + 0.07 * whole_char + 0.45 * frag_bm25 + 0.20 * frag_char + 0.20 * frag_lsa
            if method.endswith("_hierarchy"):
                score = self._hierarchy_expand(score)
        else:
            raise ValueError(f"Unknown method: {method}")

        order = np.argsort(-score)[:top_k]
        result = []
        for rank, idx in enumerate(order, start=1):
            row = self.kb[int(idx)]
            result.append({
                "rank": rank,
                "code": row["code"],
                "description": row["description"],
                "score": float(score[int(idx)]),
                "hierarchy": row.get("hierarchy", [])[-4:],
                "coding_notes": row.get("coding_notes", [])[:8],
            })
        return result


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9]+", str(text).lower()))


def classify_miss(text: str, gold_code: str, kb_by_code: dict[str, dict], retrieved_codes: set[str]) -> str:
    row = kb_by_code.get(gold_code, {})
    desc = row.get("description", "")
    text_tokens = token_set(text)
    desc_tokens = token_set(desc)

    parents = {norm_code(x.get("code", "")) for x in row.get("hierarchy", [])}
    if parents & retrieved_codes:
        return "hierarchy_near_miss"
    if normalise_space(desc).lower() and normalise_space(desc).lower() in normalise_space(text).lower():
        return "description_present_but_not_ranked"
    overlap = text_tokens & desc_tokens
    if len(overlap) >= 2:
        return "lexical_signal_ranked_low"
    if len(overlap) == 1:
        return "weak_lexical_signal"
    return "synonym_translation_or_inference_gap"


def evaluate_method(
    texts: dict[str, str],
    gold: dict[str, set[str]],
    retriever: OfflineHybridRetriever,
    method: str,
    max_k: int,
) -> tuple[dict, list[dict], list[dict]]:
    per_case: list[dict] = []
    misses: list[dict] = []
    total_gold = 0
    total_hits = {k: 0 for k in TOP_KS}
    macro_recalls = {k: [] for k in TOP_KS}
    hit_any = {k: 0 for k in TOP_KS}
    kb_by_code = {r["code"]: r for r in retriever.kb}

    for i, article_id in enumerate(sorted(texts), start=1):
        text = texts[article_id]
        g = set(gold.get(article_id, set()))
        retrieved = retriever.retrieve(text, method=method, top_k=max_k)
        ranked = [x["code"] for x in retrieved]
        total_gold += len(g)
        row = {"article_id": article_id, "n_gold": len(g), "gold_codes": "|".join(sorted(g))}

        for k in TOP_KS:
            top = set(ranked[:k])
            n_hit = len(g & top)
            total_hits[k] += n_hit
            recall = n_hit / len(g) if g else 0.0
            macro_recalls[k].append(recall)
            hit_any[k] += int(n_hit > 0)
            row[f"recall_at_{k}"] = recall
            row[f"n_hit_at_{k}"] = n_hit

        row["top35_codes"] = "|".join(ranked[:35])
        per_case.append(row)

        top35 = set(ranked[:35])
        for gc in sorted(g - top35):
            misses.append({
                "article_id": article_id,
                "gold_code": gc,
                "gold_description": kb_by_code.get(gc, {}).get("description", ""),
                "error_category": classify_miss(text, gc, kb_by_code, top35),
            })

        if i % 25 == 0:
            print(f"{method}: {i}/{len(texts)}")

    summary = {
        "method": method,
        "n_cases": len(texts),
        "n_gold_codes": total_gold,
    }
    for k in TOP_KS:
        summary[f"micro_recall_at_{k}"] = total_hits[k] / total_gold if total_gold else 0.0
        summary[f"macro_recall_at_{k}"] = float(np.mean(macro_recalls[k])) if macro_recalls[k] else 0.0
        summary[f"case_hit_rate_at_{k}"] = hit_any[k] / len(texts) if texts else 0.0
    return summary, per_case, misses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-test-eval", action="store_true", help="Required to evaluate TEST; never use TEST for tuning.")
    parser.add_argument("--limit", type=int, default=0, help="Optional deterministic first-N cases; 0 means all.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/codiesp_offline_retrieval"))
    parser.add_argument("--lsa-dim", type=int, default=128)
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test_eval:
        raise SystemExit("TEST evaluation is blocked by default. Tune on DEV; pass --allow-test-eval only for frozen final evaluation.")

    print("OFFLINE RETRIEVAL BENCHMARK: no LLM/API calls will be made.")
    texts, gold, corpus_manifest = load_codiesp_split(args.split)
    if args.limit > 0:
        keep = sorted(texts)[: args.limit]
        texts = {k: texts[k] for k in keep}
        gold = {k: gold[k] for k in keep}

    kb, kb_manifest = build_kb()
    retriever = OfflineHybridRetriever(kb, lsa_dim=args.lsa_dim)

    methods = [
        "whole_note_lexical",
        "sentence_hybrid",
        "fragment_hybrid_alias",
        "fragment_hybrid_alias_hierarchy",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    all_case_frames = []
    all_miss_frames = []
    for method in methods:
        summary, per_case, misses = evaluate_method(texts, gold, retriever, method, max(TOP_KS))
        summaries.append(summary)
        cdf = pd.DataFrame(per_case)
        cdf.insert(0, "method", method)
        all_case_frames.append(cdf)
        mdf = pd.DataFrame(misses)
        if not mdf.empty:
            mdf.insert(0, "method", method)
            all_miss_frames.append(mdf)
        print(json.dumps(summary, indent=2))

    summary_df = pd.DataFrame(summaries).sort_values(
        ["micro_recall_at_20", "micro_recall_at_35"], ascending=False
    )
    summary_df.to_csv(args.output_dir / "retrieval_summary.csv", index=False)
    pd.concat(all_case_frames, ignore_index=True).to_csv(args.output_dir / "per_case_retrieval.csv", index=False)
    if all_miss_frames:
        misses_df = pd.concat(all_miss_frames, ignore_index=True)
        misses_df.to_csv(args.output_dir / "retrieval_misses.csv", index=False)
        miss_counts = (
            misses_df.groupby(["method", "error_category"]).size().reset_index(name="n")
        )
        miss_counts.to_csv(args.output_dir / "retrieval_miss_categories.csv", index=False)

    best = summary_df.iloc[0].to_dict()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": args.split,
        "limit": args.limit,
        "selection_rule": "DEV-only: maximise micro Recall@20, tie-break by micro Recall@35",
        "selected_method": best["method"],
        "selected_metrics": best,
        "methods": methods,
        "top_ks": list(TOP_KS),
        "corpus": corpus_manifest,
        "icd_kb": kb_manifest,
        "no_llm_api_calls": True,
        "deepseek_api_key_read": False,
        "notes": [
            "fragment decomposition is deterministic, not clinical NER",
            "latent semantic channel uses offline TF-IDF + TruncatedSVD, not a downloaded embedding model",
            "TEST must remain untouched until retrieval design is frozen",
        ],
    }
    (args.output_dir / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "selected_retrieval_config.json").write_text(
        json.dumps({"selected_method": best["method"], "selection_rule": manifest["selection_rule"]}, indent=2),
        encoding="utf-8",
    )

    print("\n=== DEV RETRIEVAL SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nSelected method: {best['method']}")
    print(f"Outputs: {args.output_dir}")
    print("No DeepSeek/LLM/API calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
