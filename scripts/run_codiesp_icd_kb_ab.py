#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CODIESP_CORPUS_URL = "https://zenodo.org/records/3837305/files/codiesp.zip?download=1"
CODIESP_CODES_URL = "https://zenodo.org/records/3706838/files/codiesp_codes.zip?download=1"
CDC_CODES_URL = "https://ftp.cdc.gov/pub/health_statistics/nchs/Publications/ICD10CM/2018/2018-ICD-10-CM-Codes-File.zip"
CDC_XML_URL = "https://ftp.cdc.gov/pub/health_statistics/nchs/Publications/ICD10CM/2018/ICD-10-CM-Codes-Tables-and-Index-2018.zip"
MODEL = "deepseek-v4-pro"
SCHEMA_VERSION = "codiesp-icd-kb-ab-v0.1"

DIRECT_PROMPT_VERSION = "direct-recall-first-v0.3"
RAG_PROMPT_VERSION = "icd-kb-assisted-v0.1"


def fetch_bytes(url: str, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "med-code-icd-kb-ab/0.1"})
            with urllib.request.urlopen(req, timeout=240) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def norm_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code or "").upper())


def load_codiesp_test() -> tuple[dict[str, str], dict[str, set[str]], dict[str, str]]:
    raw = fetch_bytes(CODIESP_CORPUS_URL)
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = z.namelist()
    texts: dict[str, str] = {}
    for name in names:
        low = "/" + name.lower().lstrip("/")
        if "/test/text_files_en/" in low and low.endswith(".txt"):
            texts[Path(name).stem] = z.read(name).decode("utf-8", errors="replace")
    candidates = [n for n in names if Path(n).name.lower() == "testd.tsv"]
    if not candidates:
        raise RuntimeError("Missing testD.tsv")
    label_raw = z.read(sorted(candidates, key=len)[0]).decode("utf-8", errors="replace")
    gold: dict[str, set[str]] = defaultdict(set)
    for line in label_raw.splitlines():
        p = line.strip().split("\t")
        if len(p) >= 2:
            gold[p[0]].add(norm_code(p[1]))
    manifest = {
        "url": CODIESP_CORPUS_URL,
        "sha256": sha256(raw),
        "text_split": "test/text_files_en",
        "labels": Path(sorted(candidates, key=len)[0]).name,
    }
    return texts, dict(gold), manifest


def load_valid_codes() -> tuple[dict[str, dict], dict]:
    raw = fetch_bytes(CODIESP_CODES_URL)
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = z.namelist()
    d_files = [n for n in names if Path(n).name.lower() == "codiesp-d_codes.tsv"]
    if not d_files:
        raise RuntimeError(f"Missing codiesp-D_codes.tsv; files={names[:20]}")
    txt = z.read(d_files[0]).decode("utf-8", errors="replace")
    out: dict[str, dict] = {}
    for line in txt.splitlines():
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0].strip() and p[0].lower() != "code":
            code = norm_code(p[0])
            out[code] = {
                "code": code,
                "display_code": p[0].strip(),
                "es_description": p[1].strip(),
                "en_description": p[2].strip(),
            }
    manifest = {"url": CODIESP_CODES_URL, "sha256": sha256(raw), "file": d_files[0], "n_codes": len(out)}
    return out, manifest


def load_cdc_descriptions() -> tuple[dict[str, str], dict]:
    raw = fetch_bytes(CDC_CODES_URL)
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = [n for n in z.namelist() if n.lower().endswith(".txt")]
    if not names:
        raise RuntimeError("No CDC 2018 code description text file found")
    name = sorted(names, key=lambda n: ("codes" not in n.lower(), len(n)))[0]
    txt = z.read(name).decode("latin-1", errors="replace")
    out: dict[str, str] = {}
    for line in txt.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = re.match(r"^([A-Z][0-9A-Z]{2,6})\s+(.+)$", line)
        if m:
            out[norm_code(m.group(1))] = m.group(2).strip()
    manifest = {"url": CDC_CODES_URL, "sha256": sha256(raw), "file": name, "n_descriptions": len(out)}
    return out, manifest


def clean_text(elem: ET.Element) -> str:
    parts = [x.strip() for x in elem.itertext() if x and x.strip()]
    return " ".join(parts)


def load_cdc_xml() -> tuple[dict[str, dict], dict]:
    raw = fetch_bytes(CDC_XML_URL)
    z = zipfile.ZipFile(io.BytesIO(raw))
    xmls = [n for n in z.namelist() if n.lower().endswith(".xml")]
    tabulars = [n for n in xmls if "tabular" in Path(n).name.lower()]
    if not tabulars:
        raise RuntimeError(f"No tabular XML found; xml files={xmls[:20]}")
    name = sorted(tabulars, key=len)[0]
    root = ET.fromstring(z.read(name))
    records: dict[str, dict] = {}
    note_tags = {
        "includes", "inclusionterm", "excludes1", "excludes2", "codefirst", "codealso",
        "useadditionalcode", "note", "notes", "sevenchrnote", "sevenchrdef",
    }

    def walk_diag(diag: ET.Element, parents: list[tuple[str, str]]):
        name_el = diag.find("name")
        desc_el = diag.find("desc")
        if name_el is None:
            return
        code_display = (name_el.text or "").strip()
        code = norm_code(code_display)
        desc = clean_text(desc_el) if desc_el is not None else ""
        notes = []
        for child in list(diag):
            tag = child.tag.lower().replace("-", "")
            if tag == "diag" or tag in {"name", "desc"}:
                continue
            if tag in note_tags:
                val = clean_text(child)
                if val:
                    notes.append(f"{child.tag}: {val}")
        records[code] = {
            "code": code,
            "display_code": code_display,
            "xml_description": desc,
            "parents": [{"code": c, "description": d} for c, d in parents],
            "notes": notes,
        }
        next_parents = parents + [(code_display, desc)]
        for child in diag.findall("diag"):
            walk_diag(child, next_parents)

    for diag in root.findall(".//chapter/section/diag"):
        walk_diag(diag, [])
    # Fallback for XML variants where top-level paths differ.
    if not records:
        for diag in root.findall(".//diag"):
            parent = None
            # ElementTree has no parent pointer; fallback still preserves code/desc/notes.
            walk_diag(diag, [])
    manifest = {"url": CDC_XML_URL, "sha256": sha256(raw), "file": name, "n_xml_codes": len(records)}
    return records, manifest


def build_kb() -> tuple[list[dict], dict]:
    valid, m_valid = load_valid_codes()
    cdc_desc, m_desc = load_cdc_descriptions()
    xml, m_xml = load_cdc_xml()
    rows = []
    for code, base in valid.items():
        xr = xml.get(code, {})
        desc = cdc_desc.get(code) or base.get("en_description") or xr.get("xml_description", "")
        parents = xr.get("parents", [])
        notes = xr.get("notes", [])
        parent_text = " > ".join(f"{p['code']} {p['description']}" for p in parents[-4:])
        retrieval_text = " | ".join(x for x in [code, desc, parent_text, " ; ".join(notes)] if x)
        rows.append({
            "code": code,
            "display_code": base.get("display_code", code),
            "description": desc,
            "codiesp_en_description": base.get("en_description", ""),
            "hierarchy": parents,
            "coding_notes": notes,
            "retrieval_text": retrieval_text,
        })
    manifest = {
        "icd_version": "FY2018 ICD-10-CM",
        "task_code_space": "CodiEsp-D valid diagnosis codes (2018 version)",
        "sources": {"codiesp_code_list": m_valid, "cdc_descriptions": m_desc, "cdc_tabular_xml": m_xml},
        "n_retrievable_codes": len(rows),
    }
    return rows, manifest


class Retriever:
    def __init__(self, kb: list[dict]):
        self.kb = kb
        docs = [r["retrieval_text"] for r in kb]
        self.word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=100000)
        self.char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, max_features=140000)
        self.xw = self.word.fit_transform(docs)
        self.xc = self.char.fit_transform(docs)

    def retrieve(self, text: str, top_k: int = 35) -> list[dict]:
        full_w = cosine_similarity(self.word.transform([text]), self.xw)[0]
        full_c = cosine_similarity(self.char.transform([text]), self.xc)[0]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if len(s.strip()) >= 15]
        sentences = sentences[:100]
        if sentences:
            sw = cosine_similarity(self.word.transform(sentences), self.xw)
            sc = cosine_similarity(self.char.transform(sentences), self.xc)
            sent_w = sw.max(axis=0)
            sent_c = sc.max(axis=0)
        else:
            sent_w = np.zeros(len(self.kb))
            sent_c = np.zeros(len(self.kb))
        score = 0.25 * full_w + 0.10 * full_c + 0.45 * sent_w + 0.20 * sent_c
        order = np.argsort(-score)[:top_k]
        out = []
        for rank, idx in enumerate(order, start=1):
            r = self.kb[int(idx)]
            out.append({
                "rank": rank,
                "retrieval_score": float(score[int(idx)]),
                "code": r["code"],
                "description": r["description"],
                "hierarchy": r["hierarchy"][-4:],
                "coding_notes": r["coding_notes"][:8],
            })
        return out


def direct_prompt(case: str) -> str:
    return f"""You are a senior clinical coding specialist performing CodiEsp-D diagnosis coding with FY2018 ICD-10-CM knowledge.

PRIMARY GOAL: maximize COMPLETE recall of every diagnosis code supported by the entire clinical case while avoiding invented diagnoses. Do not code only the main diagnosis and do not stop after a short list.

Perform a careful internal multi-pass review:
1. Read the COMPLETE case sentence by sentence.
2. Enumerate every clinically established diagnosis/condition mentioned anywhere: primary diseases, chronic comorbidities, acute diagnoses, complications, metastases, infections, pathology-confirmed entities, adverse effects, and independently codable symptoms/findings when the case treats them as diagnoses.
3. Exclude procedures, tests, medications, family history, negated/ruled-out diagnoses, and pure differentials not established as present.
4. Map every retained diagnosis to the most specific FY2018 ICD-10-CM diagnosis code supported by the text. A valid answer may include 3-character parent codes when that is the supported coding level.
5. COMPLETENESS AUDIT: re-read the case and verify that every supported diagnosis mention has a code. Do not finish while a supported diagnosis remains uncoded.
6. OVER-CODING AUDIT: remove any code whose diagnosis is not directly supported by the case.

Return strict JSON only:
{{
  "codes": ["CODE1", "CODE2"],
  "accepted": [
    {{"code":"CODE1","diagnosis":"normalized diagnosis","confidence":0.0,"evidence":"short exact phrase copied from the clinical text","basis":"concise coding basis"}}
  ],
  "rejected_considered": [
    {{"code":"CODEX","confidence":0.0,"reason":"concise reason it was not selected"}}
  ]
}}

Confidence is your self-assessed confidence from 0 to 1, NOT a calibrated probability. Do not expose private chain-of-thought; provide only concise auditable basis statements.

CLINICAL CASE:
{case}"""


def rag_prompt(case: str, retrieved: list[dict]) -> str:
    knowledge = json.dumps(retrieved, ensure_ascii=False, indent=2)
    return f"""You are a senior clinical coding specialist performing CodiEsp-D diagnosis coding with FY2018 ICD-10-CM knowledge.

PRIMARY GOAL: maximize COMPLETE recall of every correct diagnosis code supported by the entire clinical case while avoiding invented diagnoses.

You are given dynamically retrieved OFFICIAL ICD-10-CM knowledge. It contains code descriptions, hierarchy context, coding notes, and retrieval scores. IMPORTANT: this retrieved list is evidence/support, NOT a whitelist and NOT a candidate restriction. You MAY add a correct code that is not in the retrieved list when the clinical text clearly supports it.

Use this process internally:
1. Read the COMPLETE case sentence by sentence and enumerate every established diagnosis/condition.
2. Use the retrieved official ICD knowledge to verify terminology, specificity, hierarchy and applicable notes. Retrieval score is relevance evidence only, not truth.
3. Map each supported diagnosis to the most specific FY2018 ICD-10-CM code supported by the case.
4. COMPLETENESS AUDIT: every supported diagnosis mention should be mapped; do not stop after only the major diagnoses.
5. KNOWLEDGE AUDIT: explicitly decide whether each retrieved code should be accepted, rejected, or remains uncertain for this case.
6. OVER-CODING AUDIT: remove diagnoses not supported by the clinical text even when retrieval ranked them highly.

Return strict JSON only:
{{
  "codes": ["CODE1", "CODE2"],
  "accepted": [
    {{"code":"CODE1","diagnosis":"normalized diagnosis","confidence":0.0,"evidence":"short exact phrase copied from the clinical text","basis":"concise basis citing text and/or retrieved ICD knowledge"}}
  ],
  "retrieved_decisions": [
    {{"code":"RETRIEVED_CODE","decision":"accept|reject|uncertain","confidence":0.0,"reason":"concise reason"}}
  ],
  "added_outside_retrieval": [
    {{"code":"CODEY","confidence":0.0,"reason":"why this code was added despite not being retrieved"}}
  ]
}}

Confidence is your self-assessed confidence from 0 to 1, NOT a calibrated probability. Do not expose private chain-of-thought; provide only concise auditable basis statements.

OFFICIAL FY2018 ICD-10-CM KNOWLEDGE RETRIEVED FOR THIS CASE:
{knowledge}

CLINICAL CASE:
{case}"""


def call_deepseek(prompt: str, api_key: str) -> tuple[dict, dict]:
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return clinically grounded ICD-10 diagnosis coding as strict JSON. Think deeply but expose only concise auditable conclusions."},
            {"role": "user", "content": prompt},
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "response_format": {"type": "json_object"},
        "max_tokens": 12000,
    }).encode("utf-8")
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as r:
                payload = json.loads(r.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
            obj = json.loads(content)
            usage = payload.get("usage", {})
            return obj, usage
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(str(last))


def score_one(g: set[str], p: set[str]) -> dict:
    tp = len(g & p); fp = len(p - g); fn = len(g - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "exact": g == p, "missed_codes_per_case": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "rag"], required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--total-cases", type=int, default=50)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--retrieval-top-k", type=int, default=35)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    texts, gold, corpus_manifest = load_codiesp_test()
    all_ids = sorted(set(texts) & set(gold))[:args.total_cases]
    ids = all_ids[args.shard_index::args.num_shards]
    if not ids:
        raise RuntimeError("Empty shard")

    retriever = None
    kb_manifest = None
    if args.mode == "rag":
        kb, kb_manifest = build_kb()
        retriever = Retriever(kb)
        (out / "icd_kb_manifest.json").write_text(json.dumps(kb_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    prompt_text = direct_prompt("[CLINICAL_TEXT]") if args.mode == "direct" else rag_prompt("[CLINICAL_TEXT]", [{"dynamic": "retrieved per case"}])
    (out / "prompt.txt").write_text(prompt_text, encoding="utf-8")

    rows = []
    failures = []
    retrieved_rows = []
    for pos, doc_id in enumerate(ids, start=1):
        retrieved = retriever.retrieve(texts[doc_id], args.retrieval_top_k) if retriever is not None else []
        prompt = direct_prompt(texts[doc_id]) if args.mode == "direct" else rag_prompt(texts[doc_id], retrieved)
        try:
            obj, usage = call_deepseek(prompt, api_key)
            pred = {norm_code(c) for c in obj.get("codes", []) if norm_code(c)}
            g = gold[doc_id]
            m = score_one(g, pred)
            accepted = obj.get("accepted", []) if isinstance(obj.get("accepted", []), list) else []
            retrieved_decisions = obj.get("retrieved_decisions", []) if isinstance(obj.get("retrieved_decisions", []), list) else []
            outside = obj.get("added_outside_retrieval", []) if isinstance(obj.get("added_outside_retrieval", []), list) else []
            rows.append({
                "article_id": doc_id,
                "mode": args.mode,
                "gold": "|".join(sorted(g)),
                "pred": "|".join(sorted(pred)),
                "accepted_json": json.dumps(accepted, ensure_ascii=False),
                "retrieved_decisions_json": json.dumps(retrieved_decisions, ensure_ascii=False),
                "added_outside_retrieval_json": json.dumps(outside, ensure_ascii=False),
                "retrieved_codes": "|".join(r["code"] for r in retrieved),
                "retrieval_gold_recall": len(g & {r["code"] for r in retrieved}) / len(g) if g else 0.0,
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "exact": m["exact"],
                "prompt_tokens": usage.get("prompt_tokens", ""), "completion_tokens": usage.get("completion_tokens", ""),
            })
            if retrieved:
                retrieved_rows.append({"article_id": doc_id, "retrieved_json": json.dumps(retrieved, ensure_ascii=False)})
        except Exception as exc:  # noqa: BLE001
            failures.append({"article_id": doc_id, "mode": args.mode, "error": str(exc)})
        print(f"{args.mode} shard={args.shard_index} {pos}/{len(ids)}", flush=True)

    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)
    if retrieved_rows:
        pd.DataFrame(retrieved_rows).to_csv(out / "retrieved_knowledge.csv", index=False)

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "model": MODEL,
        "thinking": {"type": "enabled", "reasoning_effort": "max"},
        "prompt_version": DIRECT_PROMPT_VERSION if args.mode == "direct" else RAG_PROMPT_VERSION,
        "corpus": corpus_manifest,
        "icd_knowledge": kb_manifest,
        "retrieval": None if args.mode == "direct" else {
            "top_k": args.retrieval_top_k,
            "method": "hybrid TF-IDF over official descriptions + hierarchy + XML coding notes; full-document and sentence-level similarity",
            "weights": {"full_word": 0.25, "full_char": 0.10, "sentence_word_max": 0.45, "sentence_char_max": 0.20},
            "candidate_restriction": False,
        },
        "total_cases_protocol": args.total_cases,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "case_ids": ids,
        "n_success": len(rows),
        "n_failures": len(failures),
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} requests failed")


if __name__ == "__main__":
    main()
