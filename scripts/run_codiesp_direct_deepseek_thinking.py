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

import pandas as pd

CODIESP_URL = "https://zenodo.org/records/3837305/files/codiesp.zip?download=1"
MODEL = "deepseek-v4-pro"


def fetch_bytes(url: str, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "med-code-direct-thinking/0.1"})
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
        raw = z.read(sorted(candidates, key=len)[0]).decode("utf-8", errors="replace")
        out = defaultdict(set)
        for line in raw.splitlines():
            p = line.strip().split("\t")
            if len(p) >= 2:
                out[p[0]].add(normalize_code(p[1]))
        return dict(out)

    return texts("dev"), texts("test"), labels("dev"), labels("test")


PROMPTS = {
    "direct_simple": """You are an expert ICD-10 diagnosis coding system. Read the entire clinical case and assign ALL ICD-10 DIAGNOSIS codes that are clinically supported. The goal is to recover the complete diagnosis code set, not just the main diagnosis. Include important chronic comorbidities, complications, secondary diagnoses, pathology-confirmed diagnoses, and clinically established conditions. Do not output procedure codes. Do not use any external candidate list. Return JSON only with this schema: {\"codes\":[\"CODE1\",\"CODE2\"],\"items\":[{\"code\":\"CODE1\",\"diagnosis\":\"name\",\"evidence\":\"short exact supporting phrase from the case\"}]}.

CLINICAL CASE:\n{case}""",
    "direct_exhaustive": """You are a senior clinical coding specialist performing the CodiEsp-D task: convert one complete clinical case into the most complete, exact ICD-10 DIAGNOSIS code set supported by the text. There is NO candidate list. Use your own medical and ICD-10 knowledge.

Internally perform a careful multi-pass review before answering:
1) Read the entire case, not only the opening diagnosis.
2) Enumerate every clinically established diagnosis/condition across all organ systems: primary disease, chronic comorbidities, acute complications, metastases, infections, adverse effects, relevant pathology-confirmed entities, and independently codable symptoms/findings when clearly documented as diagnoses.
3) Exclude procedures, tests, medications, ruled-out/negated diagnoses, family history, and pure differential diagnoses unless the case establishes them as present.
4) Map every retained diagnosis to the MOST SPECIFIC ICD-10 diagnosis code supported by the documentation. Prefer a specific subcode over a parent code when specificity is supported.
5) Perform an omission check: re-read the case and ask whether any diagnosis mentioned later in the narrative, past medical history, pathology, complications, or discharge assessment is still uncoded.
6) Perform an over-coding check: remove any code not directly supported by the case.

Maximize completeness while keeping every code evidence-grounded. Return JSON only, with no prose outside JSON:
{\"codes\":[\"CODE1\",\"CODE2\"],\"items\":[{\"code\":\"CODE1\",\"diagnosis\":\"normalized diagnosis name\",\"evidence\":\"short exact supporting phrase copied from the case\"}]}.

CLINICAL CASE:\n{case}""",
}


def call_deepseek(prompt: str, api_key: str) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return clinically grounded ICD-10 diagnosis coding as strict JSON. Think carefully before the final answer."},
            {"role": "user", "content": prompt},
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "response_format": {"type": "json_object"},
        "max_tokens": 6000,
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
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = json.loads(r.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
            obj = json.loads(content)
            return obj
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(str(last))


def metrics(gold: dict[str, set[str]], pred: dict[str, set[str]], ids: list[str]) -> dict:
    tp = fp = fn = exact = 0
    gold_total = pred_total = 0
    for i in ids:
        g, p = gold.get(i, set()), pred.get(i, set())
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        exact += int(g == p); gold_total += len(g); pred_total += len(p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {
        "n_documents": len(ids), "gold_codes": gold_total, "predicted_codes": pred_total,
        "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": precision, "micro_recall": recall, "micro_f1": f1, "micro_f2": f2,
        "exact_match": exact / len(ids) if ids else 0.0,
        "mean_predicted_codes_per_case": pred_total / len(ids) if ids else 0.0,
        "mean_gold_codes_per_case": gold_total / len(ids) if ids else 0.0,
    }


def run_prompt(name: str, ids: list[str], texts: dict[str, str], gold: dict[str, set[str]], api_key: str, out_csv: Path):
    pred: dict[str, set[str]] = {}
    rows = []
    failures = []
    template = PROMPTS[name]
    for index, doc_id in enumerate(ids, start=1):
        try:
            obj = call_deepseek(template.format(case=texts[doc_id]), api_key)
            codes = {normalize_code(c) for c in obj.get("codes", []) if normalize_code(c)}
            items = obj.get("items", []) if isinstance(obj.get("items", []), list) else []
            pred[doc_id] = codes
            rows.append({
                "article_id": doc_id,
                "prompt_variant": name,
                "gold": "|".join(sorted(gold.get(doc_id, set()))),
                "pred": "|".join(sorted(codes)),
                "items_json": json.dumps(items, ensure_ascii=False),
            })
        except Exception as exc:  # noqa: BLE001
            pred[doc_id] = set()
            failures.append({"article_id": doc_id, "prompt_variant": name, "error": str(exc)})
        if index % 10 == 0:
            print(f"{name}: {index}/{len(ids)}", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(failures).to_csv(out_csv.with_name(out_csv.stem + "_failures.csv"), index=False)
    m = metrics(gold, pred, ids)
    m["n_failed_requests"] = len(failures)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dev-selection-n", type=int, default=40)
    ap.add_argument("--test-n", type=int, default=250)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    dev_text, test_text, dev_gold, test_gold = load_codiesp()
    dev_ids = sorted(set(dev_text) & set(dev_gold))[:args.dev_selection_n]
    test_ids = sorted(set(test_text) & set(test_gold))[:args.test_n]

    dev_results = {}
    for name in PROMPTS:
        dev_results[name] = run_prompt(name, dev_ids, dev_text, dev_gold, api_key, out / f"dev_{name}_predictions.csv")

    # Select on development data only. Primary objective F1; recall breaks ties because the research goal prioritizes complete recovery.
    winner = sorted(PROMPTS, key=lambda n: (dev_results[n]["micro_f1"], dev_results[n]["micro_recall"]), reverse=True)[0]
    test_metrics = run_prompt(winner, test_ids, test_text, test_gold, api_key, out / "test_direct_thinking_predictions.csv")

    summary = {
        "schema_version": "codiesp-direct-deepseek-thinking-v0.1",
        "model": MODEL,
        "thinking": {"type": "enabled", "reasoning_effort": "max"},
        "no_retrieval_or_candidate_list": True,
        "dev_prompt_selection": {
            "n_documents": len(dev_ids),
            "selection_rule": "highest micro-F1 on deterministic first dev IDs; micro-recall tie-break",
            "results": dev_results,
            "selected_prompt": winner,
        },
        "test": {
            "n_documents": len(test_ids),
            "selected_prompt": winner,
            "metrics": test_metrics,
        },
        "prompts": PROMPTS,
        "scientific_note": "Prompt selection used only development cases. Test metrics are from the frozen selected prompt. No TF-IDF, similar-case retrieval, candidate code list, or external code-description lookup was provided to the model.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
