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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cohortcoder.scientific_acceptance import assess_confirmatory_recovery

NHSE_URL = (
    "https://huggingface.co/datasets/NHSEDataScience/synthetic_clinical_notes/"
    "raw/main/silver/synthetic_clinical_notes.csv"
)
SYNTHEA_URL = (
    "https://synthetichealth.github.io/synthea-sample-data/downloads/"
    "synthea_sample_data_csv_nov2021.zip"
)
CODIESP_URL = "https://zenodo.org/records/3837305/files/codiesp.zip?download=1"

TARGET_TERMS = {
    "rheumatoid_arthritis": [r"\brheumatoid arthritis\b", r"\bRA\b"],
    "hypertension": [r"\bhypertension\b", r"\bhigh blood pressure\b"],
    "depression": [r"\bdepression\b", r"\bdepressive disorder\b"],
    "diabetes_type_2": [r"\btype 2 diabetes\b", r"\btype II diabetes\b", r"\bT2DM\b"],
    "hfpef": [r"\bHFpEF\b", r"heart failure with preserved ejection fraction"],
    "vte_past": [r"\bvenous thromboembol", r"\bdeep vein thromb", r"\bpulmonary embol"],
    "obesity": [r"\bobesity\b", r"\bobese\b"],
}
COMORBIDITIES = [key for key in TARGET_TERMS if key != "rheumatoid_arthritis"]
NEGATION_RE = re.compile(r"\b(no|not|denies|denied|without|negative for|no evidence of)\b", re.I)
VALID_ASSERTIONS = {"present", "absent", "possible", "negated", "family_history", "unknown"}
VALID_TEMPORALITY = {"current", "historical", "resolved", "unknown"}


def fetch_bytes(url: str, retries: int = 4, timeout: int = 180) -> bytes:
    error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "med-code-waiting-mimic/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {error}")


def regex_positive(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text or "", flags=re.I) for pattern in patterns)


def deepseek_json(prompt: str, api_key: str, *, max_retries: int = 3) -> dict:
    body = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a strict clinical information extraction system. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    last_error = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
                text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(str(last_error))


def normalize_category(value: str) -> str:
    value = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "type_2_diabetes": "diabetes_type_2",
        "t2dm": "diabetes_type_2",
        "diabetes_mellitus_type_2": "diabetes_type_2",
        "heart_failure_with_preserved_ejection_fraction": "hfpef",
        "venous_thromboembolism": "vte_past",
        "vte": "vte_past",
        "pulmonary_embolism": "vte_past",
        "deep_vein_thrombosis": "vte_past",
    }
    return aliases.get(value, value)


def run_nhse(output_dir: Path, api_key: str | None, max_llm_notes: int) -> dict:
    df = pd.read_csv(io.BytesIO(fetch_bytes(NHSE_URL)))
    text_col = "clean_note_text"
    # Prefer unambiguous full phrase; include RA abbreviation only when needed to reach the audited RA-related set.
    full_phrase = df[text_col].fillna("").astype(str).str.contains(r"\brheumatoid arthritis\b", case=False, regex=True)
    ra_any = df[text_col].fillna("").astype(str).map(lambda x: regex_positive(x, TARGET_TERMS["rheumatoid_arthritis"]))
    ra_df = df.loc[ra_any].copy().sort_values(["person_id", "clinical_note_id"]).reset_index(drop=True)

    baseline_rows = []
    for _, row in ra_df.iterrows():
        text = str(row[text_col])
        for phenotype in COMORBIDITIES:
            baseline_rows.append(
                {
                    "clinical_note_id": row["clinical_note_id"],
                    "person_id": row["person_id"],
                    "phenotype": phenotype,
                    "keyword_positive": int(regex_positive(text, TARGET_TERMS[phenotype])),
                }
            )
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(output_dir / "nhse_keyword_baseline.csv", index=False)

    summary = {
        "n_total_notes": int(len(df)),
        "n_ra_related_notes": int(len(ra_df)),
        "n_ra_full_phrase_notes": int(full_phrase.sum()),
        "n_ra_related_patients": int(ra_df["person_id"].nunique()),
        "keyword_positive_pairs": int(baseline["keyword_positive"].sum()),
        "keyword_by_phenotype": baseline.groupby("phenotype")["keyword_positive"].sum().astype(int).to_dict(),
        "llm": {"status": "SKIPPED_NO_API_KEY"},
        "interpretation": "Synthetic-note method/evidence audit only; no clinician phenotype gold, so no clinical accuracy F1 is claimed.",
    }
    if not api_key:
        return summary

    selected = ra_df.head(max_llm_notes).copy()
    extraction_rows = []
    failures = []
    for idx, row in selected.iterrows():
        note = str(row[text_col])
        prompt = f"""Read this SYNTHETIC clinical note. Extract all clinically documented comorbidities or chronic/important co-existing diseases, excluding rheumatoid arthritis itself.
For every extracted condition return:
- condition: concise disease name
- normalized_category: one of hypertension, depression, diabetes_type_2, hfpef, vte_past, obesity, other
- evidence: one exact verbatim substring copied from the note that supports the condition
- assertion: one of present, absent, possible, negated, family_history, unknown
- temporality: one of current, historical, resolved, unknown
Do not infer diseases from medications or plausibility alone. Do not return negated/family-history items as present.
Return exactly JSON: {{"conditions": [{{...}}]}}.

NOTE:\n{note}"""
        try:
            obj = deepseek_json(prompt, api_key)
            conditions = obj.get("conditions", [])
            if not isinstance(conditions, list):
                raise ValueError("conditions is not a list")
            for item in conditions:
                condition = str(item.get("condition", "")).strip()
                category = normalize_category(item.get("normalized_category", "other"))
                evidence = str(item.get("evidence", ""))
                assertion = str(item.get("assertion", "unknown")).strip().lower()
                temporality = str(item.get("temporality", "unknown")).strip().lower()
                exact = bool(evidence) and evidence in note
                extraction_rows.append(
                    {
                        "clinical_note_id": row["clinical_note_id"],
                        "person_id": row["person_id"],
                        "condition": condition,
                        "normalized_category": category,
                        "evidence": evidence,
                        "assertion": assertion,
                        "temporality": temporality,
                        "evidence_exact_substring": int(exact),
                        "assertion_valid": int(assertion in VALID_ASSERTIONS),
                        "temporality_valid": int(temporality in VALID_TEMPORALITY),
                        "potential_negation_conflict": int(assertion == "present" and bool(NEGATION_RE.search(evidence))),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            failures.append({"clinical_note_id": row["clinical_note_id"], "error": str(exc)})

    extraction = pd.DataFrame(extraction_rows)
    extraction.to_csv(output_dir / "nhse_open_ended_llm_extractions.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "nhse_llm_failures.csv", index=False)

    positive_assertions = {"present"}
    positive = extraction[extraction["assertion"].isin(positive_assertions)] if len(extraction) else extraction
    llm_pairs = set()
    if len(positive):
        for _, r in positive.iterrows():
            if r["normalized_category"] in COMORBIDITIES:
                llm_pairs.add((str(r["clinical_note_id"]), r["normalized_category"]))
    keyword_pairs = {
        (str(r["clinical_note_id"]), r["phenotype"])
        for _, r in baseline[baseline["keyword_positive"] == 1].iterrows()
        if str(r["clinical_note_id"]) in set(selected["clinical_note_id"].astype(str))
    }
    overlap = len(keyword_pairs & llm_pairs)

    review = extraction[
        (extraction.get("evidence_exact_substring", pd.Series(dtype=int)) == 0)
        | (extraction.get("potential_negation_conflict", pd.Series(dtype=int)) == 1)
        | (extraction.get("assertion_valid", pd.Series(dtype=int)) == 0)
        | (extraction.get("temporality_valid", pd.Series(dtype=int)) == 0)
    ] if len(extraction) else extraction
    review.to_csv(output_dir / "nhse_error_review_candidates.csv", index=False)

    summary["llm"] = {
        "status": "COMPLETED",
        "n_notes_attempted": int(len(selected)),
        "n_notes_failed": len(failures),
        "n_extracted_items": int(len(extraction)),
        "n_present_items": int(len(positive)),
        "n_unique_condition_strings_present": int(positive["condition"].str.lower().nunique()) if len(positive) else 0,
        "evidence_exact_substring_rate": float(extraction["evidence_exact_substring"].mean()) if len(extraction) else None,
        "valid_assertion_rate": float(extraction["assertion_valid"].mean()) if len(extraction) else None,
        "valid_temporality_rate": float(extraction["temporality_valid"].mean()) if len(extraction) else None,
        "potential_negation_conflicts": int(extraction["potential_negation_conflict"].sum()) if len(extraction) else 0,
        "automatic_error_review_candidates": int(len(review)),
        "keyword_positive_pairs_in_llm_subset": len(keyword_pairs),
        "llm_present_target_pairs": len(llm_pairs),
        "keyword_llm_pair_overlap": overlap,
        "keyword_pair_coverage_by_llm": overlap / len(keyword_pairs) if keyword_pairs else None,
        "note": "Keyword/LLM agreement is not accuracy because NHSE Silver has no clinician phenotype gold.",
    }
    if len(positive):
        top = positive["condition"].str.lower().value_counts().head(30)
        top.rename_axis("condition").reset_index(name="count").to_csv(output_dir / "nhse_top_open_ended_conditions.csv", index=False)
    return summary


def deterministic_uniform(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def run_synthea(output_dir: Path) -> dict:
    archive = zipfile.ZipFile(io.BytesIO(fetch_bytes(SYNTHEA_URL)))
    names = archive.namelist()
    def member(suffix: str) -> str:
        for name in names:
            if name.endswith("/" + suffix) or name == suffix:
                return name
        raise KeyError(suffix)
    conditions = pd.read_csv(archive.open(member("conditions.csv")))
    patients = pd.read_csv(archive.open(member("patients.csv")))
    desc_col = "DESCRIPTION" if "DESCRIPTION" in conditions.columns else "Description"
    patient_col = "PATIENT" if "PATIENT" in conditions.columns else "Patient"
    patient_ids = patients["Id" if "Id" in patients.columns else "ID"].astype(str).tolist()
    desc = conditions[desc_col].fillna("").astype(str)

    truth = defaultdict(lambda: {p: 0 for p in TARGET_TERMS})
    for pid, text in zip(conditions[patient_col].astype(str), desc):
        for phenotype, patterns in TARGET_TERMS.items():
            if regex_positive(text, patterns):
                truth[pid][phenotype] = 1

    # Controlled C4 mechanics: full structured truth is G. C is made incomplete by deterministic masking.
    # T_oracle represents a perfectly validated text extractor; T_imperfect is a fixed, prespecified synthetic detector.
    rates = [0.10, 0.20, 0.30, 0.50]
    experiment_rows = []
    summaries = {}
    for rate in rates:
        rows_oracle = []
        rows_imperfect = []
        injected = 0
        positive_pairs = 0
        for pid in patient_ids:
            vals = truth[str(pid)]
            for phenotype in TARGET_TERMS:
                g = int(vals[phenotype])
                if g:
                    positive_pairs += 1
                masked = g == 1 and deterministic_uniform(str(pid), phenotype, str(rate), "mask") < rate
                c = 0 if masked else g
                if masked:
                    injected += 1
                t_oracle = g
                if g:
                    t_imperfect = int(deterministic_uniform(str(pid), phenotype, "sens") < 0.85)
                else:
                    t_imperfect = int(deterministic_uniform(str(pid), phenotype, "fpr") < 0.02)
                base = {"subject_id": str(pid), "phenotype": phenotype, "gold": g, "structured": c}
                rows_oracle.append({**base, "text": t_oracle})
                rows_imperfect.append({**base, "text": t_imperfect})
                experiment_rows.append({"missingness_rate": rate, **base, "text_oracle": t_oracle, "text_imperfect": t_imperfect, "injected_missing": int(masked)})

        oracle_result = assess_confirmatory_recovery(
            rows=rows_oracle,
            stage12_final_status="PASS",
            structured_scope_validated=True,
            complete_pair_coverage=True,
            n_bootstrap=500,
            seed=20260723,
        )
        imperfect_result = assess_confirmatory_recovery(
            rows=rows_imperfect,
            stage12_final_status="PASS",
            structured_scope_validated=True,
            complete_pair_coverage=True,
            n_bootstrap=500,
            seed=20260723,
        )
        key = f"{int(rate*100)}pct"
        summaries[key] = {
            "injected_missing_gold_positive_pairs": injected,
            "all_gold_positive_pairs": positive_pairs,
            "realized_missing_fraction": injected / positive_pairs if positive_pairs else None,
            "oracle_text": oracle_result,
            "prespecified_imperfect_text": imperfect_result,
            "imperfect_text_simulation_parameters": {"sensitivity": 0.85, "false_positive_rate": 0.02},
        }
    pd.DataFrame(experiment_rows).to_csv(output_dir / "synthea_known_missingness_rows.csv", index=False)
    return {
        "n_patients": len(patient_ids),
        "n_condition_rows": int(len(conditions)),
        "target_positive_patient_pairs": {p: sum(truth[pid][p] for pid in truth) for p in TARGET_TERMS},
        "missingness_scenarios": summaries,
        "interpretation": "Controlled methods validation only. Missingness is injected by design; recovery estimates validate C4 mechanics and must not be interpreted as real-world under-recording prevalence.",
    }


def normalize_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code).upper())


def read_codiesp() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], list[str]]:
    z = zipfile.ZipFile(io.BytesIO(fetch_bytes(CODIESP_URL)))
    names = z.namelist()
    lower_map = {n.lower(): n for n in names}

    def texts(split: str) -> dict[str, str]:
        result = {}
        token = f"/{split}/text_files_en/"
        for name in names:
            low = "/" + name.lower().lstrip("/")
            if token in low and low.endswith(".txt"):
                result[Path(name).stem] = z.read(name).decode("utf-8", errors="replace")
        return result

    def labels(split: str) -> dict[str, set[str]]:
        candidates = []
        exact = f"{split}d.tsv"
        for name in names:
            base = Path(name).name.lower()
            if base == exact:
                candidates.append(name)
        if not candidates:
            for name in names:
                base = Path(name).name.lower()
                if split in name.lower() and base.endswith("d.tsv"):
                    candidates.append(name)
        if not candidates:
            raise RuntimeError(f"Could not find diagnosis annotation TSV for {split}; sample names={names[:80]}")
        raw = z.read(sorted(candidates, key=len)[0]).decode("utf-8", errors="replace")
        out = defaultdict(set)
        for line in raw.splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                out[parts[0]].add(normalize_code(parts[1]))
        return dict(out)

    return texts("train"), texts("dev"), texts("test"), labels("train"), labels("dev"), labels("test"), names


def rank_codes(train_ids: list[str], train_labels: dict[str, set[str]], similarities: np.ndarray, top_docs: int = 15) -> list[str]:
    order = np.argsort(-similarities)[:top_docs]
    scores = defaultdict(float)
    best_rank = {}
    for rank, idx in enumerate(order):
        sim = float(similarities[idx])
        for code in train_labels.get(train_ids[idx], set()):
            scores[code] += max(sim, 0.0)
            best_rank[code] = min(best_rank.get(code, rank), rank)
    return sorted(scores, key=lambda c: (-scores[c], best_rank[c], c))


def multilabel_metrics(gold: dict[str, set[str]], pred: dict[str, set[str]], ids: list[str]) -> dict:
    tp = fp = fn = exact = 0
    gold_total = 0
    pred_total = 0
    for doc_id in ids:
        g = gold.get(doc_id, set())
        p = pred.get(doc_id, set())
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        gold_total += len(g); pred_total += len(p)
        exact += int(g == p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_documents": len(ids), "gold_codes": gold_total, "predicted_codes": pred_total,
        "micro_precision": precision, "micro_recall": recall, "micro_f1": f1,
        "exact_match": exact / len(ids) if ids else 0.0,
    }


def run_codiesp(output_dir: Path, api_key: str | None, deepseek_n: int) -> dict:
    train_texts, dev_texts, test_texts, train_gold, dev_gold, test_gold, names = read_codiesp()
    train_ids = sorted(set(train_texts) & set(train_gold))
    dev_ids = sorted(set(dev_texts) & set(dev_gold))
    test_ids = sorted(set(test_texts) & set(test_gold))
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
    x_train = vectorizer.fit_transform([train_texts[i] for i in train_ids])
    x_dev = vectorizer.transform([dev_texts[i] for i in dev_ids])
    x_test = vectorizer.transform([test_texts[i] for i in test_ids])

    def ranked_for(matrix) -> dict[str, list[str]]:
        sims = cosine_similarity(matrix, x_train)
        ids = dev_ids if matrix.shape[0] == len(dev_ids) else test_ids
        return {doc_id: rank_codes(train_ids, train_gold, sims[j], top_docs=15) for j, doc_id in enumerate(ids)}

    dev_ranked = ranked_for(x_dev)
    test_ranked = ranked_for(x_test)
    dev_scores = []
    for top_n in range(1, 11):
        pred = {i: set(dev_ranked[i][:top_n]) for i in dev_ids}
        m = multilabel_metrics(dev_gold, pred, dev_ids)
        dev_scores.append({"top_n": top_n, **m})
    dev_df = pd.DataFrame(dev_scores)
    best_top_n = int(dev_df.sort_values(["micro_f1", "micro_recall", "top_n"], ascending=[False, False, True]).iloc[0]["top_n"])
    dev_df.to_csv(output_dir / "codiesp_dev_topn_selection.csv", index=False)

    baseline_pred = {i: set(test_ranked[i][:best_top_n]) for i in test_ids}
    baseline_metrics = multilabel_metrics(test_gold, baseline_pred, test_ids)
    candidate_recall_num = sum(len(test_gold[i] & set(test_ranked[i][:20])) for i in test_ids)
    candidate_recall_den = sum(len(test_gold[i]) for i in test_ids)
    baseline_metrics.update({
        "selected_top_n_from_dev": best_top_n,
        "candidate_recall_at_20": candidate_recall_num / candidate_recall_den if candidate_recall_den else 0.0,
    })
    pd.DataFrame(
        [{"article_id": i, "gold": "|".join(sorted(test_gold[i])), "pred": "|".join(sorted(baseline_pred[i])), "top20": "|".join(test_ranked[i][:20])} for i in test_ids]
    ).to_csv(output_dir / "codiesp_tfidf_test_predictions.csv", index=False)

    result = {
        "dataset": "CodiEsp v1.4 official release",
        "english_text": "Officially distributed machine-translated English text_files_en; sentence-split by the translation process.",
        "n_train_texts_with_diagnosis_gold": len(train_ids),
        "n_dev_texts_with_diagnosis_gold": len(dev_ids),
        "n_test_texts_with_diagnosis_gold": len(test_ids),
        "tfidf_retrieval_baseline": baseline_metrics,
        "deepseek_candidate_assisted": {"status": "SKIPPED_NO_API_KEY"},
        "scientific_role": "Real gold-standard Clinical text -> ICD-10 diagnosis coding external benchmark. It is Spanish clinical-case data with official machine-translated English, not MIMIC EHR discharge notes.",
    }
    if not api_key:
        return result

    subset = test_ids[:deepseek_n]
    ds_pred = {}
    rows = []
    failures = []
    for doc_id in subset:
        candidates = test_ranked[doc_id][:20]
        prompt = f"""Assign ICD-10 diagnosis codes to this machine-translated English clinical case from the CodiEsp diagnosis-coding benchmark.
Return diagnosis codes only, not procedures. Candidate codes retrieved from similar labelled training cases are provided as hints; select only codes supported by the case, and you may add another ICD-10 diagnosis code only when clearly necessary.
Return exactly JSON: {{"codes": ["CODE1", "CODE2"]}}.
Normalize to specific ICD-10/ICD-10-CM diagnosis codes when supported.

CANDIDATE CODES:\n{', '.join(candidates)}\n\nCLINICAL CASE:\n{test_texts[doc_id]}"""
        try:
            obj = deepseek_json(prompt, api_key)
            codes = {normalize_code(c) for c in obj.get("codes", []) if normalize_code(c)}
            ds_pred[doc_id] = codes
            rows.append({"article_id": doc_id, "gold": "|".join(sorted(test_gold[doc_id])), "pred": "|".join(sorted(codes)), "candidate_top20": "|".join(candidates)})
        except Exception as exc:  # noqa: BLE001
            failures.append({"article_id": doc_id, "error": str(exc)})
            ds_pred[doc_id] = set()
    pd.DataFrame(rows).to_csv(output_dir / "codiesp_deepseek_predictions.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "codiesp_deepseek_failures.csv", index=False)
    metrics = multilabel_metrics(test_gold, ds_pred, subset)
    cand_num = sum(len(test_gold[i] & set(test_ranked[i][:20])) for i in subset)
    cand_den = sum(len(test_gold[i]) for i in subset)
    metrics.update({
        "n_failed_requests": len(failures),
        "candidate_recall_at_20_on_subset": cand_num / cand_den if cand_den else 0.0,
        "subset_selection": "first article IDs in deterministic sorted test order; no test tuning",
    })
    result["deepseek_candidate_assisted"] = {"status": "COMPLETED", **metrics}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nhse-max-llm-notes", type=int, default=97)
    parser.add_argument("--codiesp-deepseek-n", type=int, default=50)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or None

    summary = {
        "schema_version": "waiting-mimic-experiments-v0.1",
        "nhse_ra_experiment": run_nhse(out, api_key, args.nhse_max_llm_notes),
        "synthea_known_missingness_c4": run_synthea(out),
        "codiesp_clinical_text_to_icd10": run_codiesp(out, api_key, args.codiesp_deepseek_n),
        "scientific_boundaries": {
            "nhse_can_validate_pipeline_and_evidence_contract": True,
            "nhse_can_replace_authorised_mipa_c3": False,
            "synthea_can_validate_c4_mechanics_with_known_injected_missingness": True,
            "synthea_can_estimate_real_underrecording_prevalence": False,
            "codiesp_can_provide_real_gold_icd10_coding_metrics": True,
            "codiesp_can_replace_ra_specific_mipa_c3_c4": False,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
