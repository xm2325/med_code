#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CODIESP_URL = "https://zenodo.org/records/3837305/files/codiesp.zip?download=1"


def fetch_bytes(url: str, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "med-code-codiesp/0.1"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(str(last))


def normalize_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code).upper())


def load_codiesp():
    z = zipfile.ZipFile(io.BytesIO(fetch_bytes(CODIESP_URL)))
    names = z.namelist()

    def texts(split: str) -> dict[str, str]:
        out = {}
        token = f"/{split}/text_files_en/"
        for name in names:
            low = "/" + name.lower().lstrip("/")
            if token in low and low.endswith(".txt"):
                out[Path(name).stem] = z.read(name).decode("utf-8", errors="replace")
        return out

    def labels(split: str) -> dict[str, set[str]]:
        exact = f"{split}d.tsv"
        candidates = [n for n in names if Path(n).name.lower() == exact]
        if not candidates:
            raise RuntimeError(f"Missing {exact}; TSVs={[n for n in names if n.lower().endswith('.tsv')][:20]}")
        raw = z.read(sorted(candidates, key=len)[0]).decode("utf-8", errors="replace")
        out = defaultdict(set)
        for line in raw.splitlines():
            p = line.strip().split("\t")
            if len(p) >= 2:
                out[p[0]].add(normalize_code(p[1]))
        return dict(out)

    return texts("train"), texts("dev"), texts("test"), labels("train"), labels("dev"), labels("test")


def rank_codes(train_ids, train_gold, similarities, top_docs=15):
    order = np.argsort(-similarities)[:top_docs]
    score = defaultdict(float)
    first = {}
    for rank, idx in enumerate(order):
        sim = max(float(similarities[idx]), 0.0)
        for code in train_gold.get(train_ids[idx], set()):
            score[code] += sim
            first[code] = min(first.get(code, rank), rank)
    return sorted(score, key=lambda c: (-score[c], first[c], c))


def metrics(gold, pred, ids):
    tp = fp = fn = exact = 0
    for i in ids:
        g, p = gold.get(i, set()), pred.get(i, set())
        tp += len(g & p); fp += len(p - g); fn += len(g - p); exact += int(g == p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_documents": len(ids), "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": precision, "micro_recall": recall, "micro_f1": f1,
        "exact_match": exact / len(ids) if ids else 0.0,
    }


def deepseek_json(prompt: str, api_key: str) -> dict:
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a strict clinical coding system. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode()
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions", data=body,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=180) as r:
                payload = json.loads(r.read().decode())
            text = payload["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001
            last = exc; time.sleep(2**attempt)
    raise RuntimeError(str(last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--deepseek-n", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train_text, dev_text, test_text, train_gold, dev_gold, test_gold = load_codiesp()
    train_ids = sorted(set(train_text) & set(train_gold))
    dev_ids = sorted(set(dev_text) & set(dev_gold))
    test_ids = sorted(set(test_text) & set(test_gold))

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_features=60000, sublinear_tf=True)
    xtr = vec.fit_transform([train_text[i] for i in train_ids])
    xdev = vec.transform([dev_text[i] for i in dev_ids])
    xtest = vec.transform([test_text[i] for i in test_ids])

    def ranked(matrix, ids):
        sims = cosine_similarity(matrix, xtr)
        return {doc_id: rank_codes(train_ids, train_gold, sims[j]) for j, doc_id in enumerate(ids)}

    dev_rank = ranked(xdev, dev_ids)
    test_rank = ranked(xtest, test_ids)
    selection = []
    for n in range(1,11):
        m = metrics(dev_gold, {i:set(dev_rank[i][:n]) for i in dev_ids}, dev_ids)
        selection.append({"top_n":n, **m})
    sel = pd.DataFrame(selection)
    best_n = int(sel.sort_values(["micro_f1","micro_recall","top_n"], ascending=[False,False,True]).iloc[0].top_n)
    sel.to_csv(out/"dev_topn_selection.csv", index=False)

    pred = {i:set(test_rank[i][:best_n]) for i in test_ids}
    base = metrics(test_gold, pred, test_ids)
    denom = sum(len(test_gold[i]) for i in test_ids)
    base["candidate_recall_at_20"] = sum(len(test_gold[i] & set(test_rank[i][:20])) for i in test_ids)/denom
    base["selected_top_n_from_dev"] = best_n
    pd.DataFrame([{"article_id":i,"gold":"|".join(sorted(test_gold[i])),"pred":"|".join(sorted(pred[i])),"top20":"|".join(test_rank[i][:20])} for i in test_ids]).to_csv(out/"tfidf_test_predictions.csv",index=False)

    api_key = os.getenv("DEEPSEEK_API_KEY","").strip()
    ds = {"status":"SKIPPED_NO_API_KEY"}
    if api_key:
        subset = test_ids[:args.deepseek_n]
        ds_pred = {}; rows=[]; failures=[]
        for i in subset:
            candidates = test_rank[i][:20]
            prompt = f"""Code this machine-translated English clinical case with ICD-10 DIAGNOSIS codes for the CodiEsp-D task. Do not return procedure codes.
Candidate diagnosis codes retrieved from similar labelled training cases: {', '.join(candidates)}
Choose only diagnoses clinically supported by the case. You may add a non-candidate code only if clearly necessary.
Return exactly {{"codes":["CODE1","CODE2"]}}.

CASE:\n{test_text[i]}"""
            try:
                obj=deepseek_json(prompt,api_key)
                codes={normalize_code(x) for x in obj.get("codes",[]) if normalize_code(x)}
                ds_pred[i]=codes
                rows.append({"article_id":i,"gold":"|".join(sorted(test_gold[i])),"pred":"|".join(sorted(codes)),"top20":"|".join(candidates)})
            except Exception as exc:  # noqa: BLE001
                ds_pred[i]=set(); failures.append({"article_id":i,"error":str(exc)})
        pd.DataFrame(rows).to_csv(out/"deepseek_predictions.csv",index=False)
        pd.DataFrame(failures).to_csv(out/"deepseek_failures.csv",index=False)
        ds={"status":"COMPLETED",**metrics(test_gold,ds_pred,subset),"n_failed_requests":len(failures)}
        d=sum(len(test_gold[i]) for i in subset)
        ds["candidate_recall_at_20_on_subset"]=sum(len(test_gold[i]&set(test_rank[i][:20])) for i in subset)/d if d else 0
        ds["subset_selection"]="first sorted test article IDs; no test tuning"

    summary={
        "schema_version":"codiesp-icd10-benchmark-v0.1",
        "dataset":"CodiEsp v1.4",
        "n_train":len(train_ids),"n_dev":len(dev_ids),"n_test":len(test_ids),
        "text_language":"officially distributed machine-translated English text_files_en",
        "tfidf_retrieval_baseline":base,
        "deepseek_candidate_assisted":ds,
        "interpretation":"Real gold-standard external Clinical text -> ICD-10 diagnosis benchmark; not an RA EHR under-recording dataset and not a substitute for MIPA/MIMIC C3/C4."
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
