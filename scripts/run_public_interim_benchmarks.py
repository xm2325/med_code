#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

TARGET_TERMS = {
    "rheumatoid_arthritis": [r"\brheumatoid arthritis\b", r"\bRA\b"],
    "hypertension": [r"\bhypertension\b", r"\bhigh blood pressure\b"],
    "depression": [r"\bdepression\b", r"\bdepressive disorder\b"],
    "diabetes_type_2": [r"\btype 2 diabetes\b", r"\btype II diabetes\b", r"\bT2DM\b"],
    "hfpef": [r"\bHFpEF\b", r"heart failure with preserved ejection fraction"],
    "vte_past": [r"\bvenous thromboembol", r"\bdeep vein thromb", r"\bpulmonary embol"],
    "obesity": [r"\bobesity\b", r"\bobese\b"],
}

SIMSUM_FEATURES = {
    "dysp": [r"\bdyspn(?:ea|oea)\b", r"shortness of breath", r"\bSOB\b"],
    "cough": [r"\bcough(?:ing)?\b"],
    "pain": [r"\bpain\b", r"\bache\b"],
    "fever": [r"\bfever\b", r"\bfebrile\b", r"\bpyrexia\b"],
    "nasal": [r"\bnasal\b", r"runny nose", r"rhinorrh(?:ea|oea)", r"\bcongestion\b"],
    "asthma": [r"\basthma\b", r"\basthmatic\b"],
    "smoking": [r"\bsmok(?:e|er|ing|es|ed)\b", r"\btobacco\b"],
    "COPD": [r"\bCOPD\b", r"chronic obstructive pulmonary"],
    "hay_fever": [r"hay fever", r"allergic rhinitis"],
}


def _fetch_bytes(url: str, retries: int = 4, timeout: int = 90) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "med-code-public-benchmark/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def _fetch_json(url: str) -> dict:
    return json.loads(_fetch_bytes(url).decode("utf-8"))


def _regex_positive(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text or "", flags=re.IGNORECASE) for pattern in patterns)


def _binary(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "present"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n", "absent"}:
        return 0
    try:
        return int(float(text) > 0)
    except ValueError as exc:
        raise ValueError(f"Cannot parse binary label: {value!r}") from exc


def _classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "n": len(y_true),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
    }


def _text_audit(name: str, rows: list[dict], text_key: str, id_key: str | None = None) -> dict:
    texts = [str(row.get(text_key, "") or "") for row in rows]
    mention_counts = {
        phenotype: sum(_regex_positive(text, patterns) for text in texts)
        for phenotype, patterns in TARGET_TERMS.items()
    }
    lengths = [len(text) for text in texts]
    unique_ids = None
    if id_key:
        unique_ids = len({str(row.get(id_key)) for row in rows if row.get(id_key) is not None})
    return {
        "dataset": name,
        "n_rows_audited": len(rows),
        "n_unique_ids": unique_ids,
        "mean_text_characters": sum(lengths) / len(lengths) if lengths else 0.0,
        "median_text_characters": float(pd.Series(lengths).median()) if lengths else 0.0,
        "target_explicit_mention_counts": mention_counts,
        "limitations": "Lexical mention audit only; no clinician gold phenotype accuracy is claimed.",
    }


def audit_nhse() -> dict:
    notes_url = (
        "https://huggingface.co/datasets/NHSEDataScience/synthetic_clinical_notes/"
        "raw/main/silver/synthetic_clinical_notes.csv"
    )
    data = pd.read_csv(io.BytesIO(_fetch_bytes(notes_url)))
    required = {"clean_note_text", "person_id", "clinical_note_id", "note_type"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"NHSE notes missing columns: {missing}")
    rows = data.to_dict("records")
    result = _text_audit("NHSE synthetic clinical notes (Silver)", rows, "clean_note_text", "person_id")
    result.update(
        {
            "n_notes": int(len(data)),
            "n_patients": int(data["person_id"].nunique()),
            "n_note_types": int(data["note_type"].nunique()),
            "top_note_types": data["note_type"].astype(str).value_counts().head(10).to_dict(),
            "scientific_role": "Longitudinal note-pipeline and evidence-contract stress test; not a C3 clinical accuracy benchmark because released Silver data lack clinician phenotype gold labels.",
        }
    )
    return result


def _hf_rows(dataset: str, offsets: list[int], length: int = 100) -> list[dict]:
    rows: list[dict] = []
    for offset in offsets:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": length,
            }
        )
        payload = _fetch_json(f"https://datasets-server.huggingface.co/rows?{query}")
        rows.extend(item["row"] for item in payload.get("rows", []))
    return rows


def audit_asclepius() -> dict:
    offsets = [0, 5000, 20000, 50000, 100000, 150000]
    rows = _hf_rows("starmpcc/Asclepius-Synthetic-Clinical-Notes", offsets)
    result = _text_audit("Asclepius Synthetic Clinical Notes", rows, "note", "patient_id")
    result.update(
        {
            "sampling": {"offsets": offsets, "rows_per_offset": 100},
            "task_distribution": dict(Counter(str(row.get("task", "")) for row in rows)),
            "scientific_role": "Large public synthetic discharge-note stress set for NLP/prompt robustness; question-answer labels are generated instructions, not clinician phenotype gold for RA comorbidity C3/C4.",
        }
    )
    return result


def audit_pmc_patients() -> dict:
    # Deterministic spread across the published 167k scale rather than taking only the first rows.
    offsets = [0, 18000, 36000, 54000, 72000, 90000, 108000, 126000, 144000, 160000]
    last_error = None
    rows = []
    for dataset in ["THUMedInfo/PMC-Patients", "aisc-team-b1/PMC-Patients"]:
        try:
            rows = _hf_rows(dataset, offsets)
            if rows:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    if not rows:
        raise RuntimeError(f"Could not fetch PMC-Patients sample: {last_error}")
    text_key = "patient" if "patient" in rows[0] else "note"
    id_key = "patient_id" if "patient_id" in rows[0] else None
    result = _text_audit("PMC-Patients", rows, text_key, id_key)
    result.update(
        {
            "sampling": {"offsets": offsets, "rows_per_offset": 100},
            "scientific_role": "Real medical case-report patient summaries for open-ended extraction/domain-shift testing; not EHR notes and no matched structured ICD gold, so not confirmatory C4.",
        }
    )
    return result


def audit_synthea() -> dict:
    url = (
        "https://synthetichealth.github.io/synthea-sample-data/downloads/"
        "synthea_sample_data_csv_nov2021.zip"
    )
    archive = zipfile.ZipFile(io.BytesIO(_fetch_bytes(url)))
    names = archive.namelist()

    def member(name: str) -> str:
        matches = [candidate for candidate in names if candidate.endswith("/" + name) or candidate == name]
        if not matches:
            raise KeyError(f"Missing {name} in Synthea archive")
        return matches[0]

    conditions = pd.read_csv(archive.open(member("conditions.csv")))
    patients = pd.read_csv(archive.open(member("patients.csv")))
    description_col = "DESCRIPTION" if "DESCRIPTION" in conditions.columns else "Description"
    patient_col = "PATIENT" if "PATIENT" in conditions.columns else "Patient"
    descriptions = conditions[description_col].fillna("").astype(str)
    mention_counts = {
        phenotype: int(descriptions.map(lambda text: _regex_positive(text, patterns)).sum())
        for phenotype, patterns in TARGET_TERMS.items()
    }
    patient_counts = {
        phenotype: int(
            conditions.loc[descriptions.map(lambda text: _regex_positive(text, patterns)), patient_col].nunique()
        )
        for phenotype, patterns in TARGET_TERMS.items()
    }
    ra_mask = descriptions.map(lambda text: _regex_positive(text, TARGET_TERMS["rheumatoid_arthritis"]))
    ra_patients = set(conditions.loc[ra_mask, patient_col].astype(str))
    comorbidity_by_ra_patient = {}
    for phenotype, patterns in TARGET_TERMS.items():
        if phenotype == "rheumatoid_arthritis":
            continue
        phenotype_patients = set(
            conditions.loc[
                descriptions.map(lambda text: _regex_positive(text, patterns)), patient_col
            ].astype(str)
        )
        comorbidity_by_ra_patient[phenotype] = len(ra_patients & phenotype_patients)
    return {
        "dataset": "Synthea sample CSV Nov 2021",
        "n_patients": int(len(patients)),
        "n_condition_rows": int(len(conditions)),
        "target_condition_row_counts": mention_counts,
        "target_unique_patient_counts": patient_counts,
        "n_ra_patients": len(ra_patients),
        "ra_patient_comorbidity_counts": comorbidity_by_ra_patient,
        "scientific_role": "Structured-EHR and cohort/discordance mechanics test. Standard sample CSV has structured conditions but no independently authored free-text phenotype gold; cannot establish real text-vs-code under-recording.",
    }


def _load_simsum() -> pd.DataFrame:
    urls = [
        "https://huggingface.co/datasets/prabaey/simsum/resolve/main/SimSUM.csv",
        "https://huggingface.co/datasets/prabaey/simsum/raw/main/SimSUM.csv",
    ]
    errors = []
    for url in urls:
        try:
            data = _fetch_bytes(url)
            return pd.read_csv(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not load SimSUM: " + " | ".join(errors))


def _evaluate_regex_simsum(df: pd.DataFrame, text_col: str) -> tuple[dict, pd.DataFrame]:
    rows = []
    for feature, patterns in SIMSUM_FEATURES.items():
        if feature not in df.columns:
            continue
        y_true = [_binary(value) for value in df[feature].tolist()]
        y_pred = [_regex_positive(str(text or ""), patterns) for text in df[text_col].fillna("").tolist()]
        metrics = _classification_metrics(y_true, [int(value) for value in y_pred])
        rows.append({"feature": feature, "text_column": text_col, **metrics})
    metrics_df = pd.DataFrame(rows)
    return {
        "n_records": int(len(df)),
        "n_features": int(len(metrics_df)),
        "macro_f1": float(metrics_df["f1"].mean()) if len(metrics_df) else None,
        "macro_precision": float(metrics_df["precision"].mean()) if len(metrics_df) else None,
        "macro_recall": float(metrics_df["recall"].mean()) if len(metrics_df) else None,
        "macro_specificity": float(metrics_df["specificity"].mean()) if len(metrics_df) else None,
    }, metrics_df


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


def _deepseek_predict(note: str, api_key: str) -> dict[str, int]:
    labels = list(SIMSUM_FEATURES)
    prompt = (
        "Read the synthetic clinical note and classify whether each clinical feature is PRESENT in the patient. "
        "Do not infer a feature merely because it is medically plausible. Return exactly one JSON object with keys: "
        + ", ".join(labels)
        + ". Each value must be 0 or 1.\n\nNOTE:\n"
        + note
    )
    body = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a strict clinical information extraction classifier."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = _parse_json_object(content)
    return {label: _binary(parsed[label]) for label in labels}


def evaluate_deepseek_simsum(df: pd.DataFrame, sample_size: int, output_dir: Path) -> dict:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return {"status": "SKIPPED_NO_API_KEY"}
    if sample_size < 1:
        return {"status": "SKIPPED_SAMPLE_SIZE_ZERO"}
    sample = df.sample(n=min(sample_size, len(df)), random_state=20260723).reset_index(drop=True)
    predictions = []
    errors = []
    for idx, row in sample.iterrows():
        try:
            pred = _deepseek_predict(str(row["advanced_text"]), api_key)
            predictions.append({"row": idx, **pred})
        except Exception as exc:  # noqa: BLE001
            errors.append({"row": idx, "error": str(exc)})
        time.sleep(0.12)
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_dir / "simsum_deepseek_predictions.csv", index=False)
    pd.DataFrame(errors).to_csv(output_dir / "simsum_deepseek_errors.csv", index=False)
    if pred_df.empty:
        return {"status": "FAILED_NO_VALID_PREDICTIONS", "n_errors": len(errors)}

    metric_rows = []
    valid_indices = pred_df["row"].astype(int).tolist()
    for feature in SIMSUM_FEATURES:
        y_true = [_binary(sample.loc[index, feature]) for index in valid_indices]
        y_pred = [int(value) for value in pred_df[feature].tolist()]
        metric_rows.append({"feature": feature, **_classification_metrics(y_true, y_pred)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "simsum_deepseek_metrics.csv", index=False)
    return {
        "status": "COMPLETED",
        "model": "deepseek-chat",
        "text_column": "advanced_text",
        "n_requested": int(len(sample)),
        "n_valid_predictions": int(len(pred_df)),
        "n_errors": len(errors),
        "coverage": float(len(pred_df) / len(sample)),
        "macro_f1": float(metrics["f1"].mean()),
        "macro_precision": float(metrics["precision"].mean()),
        "macro_recall": float(metrics["recall"].mean()),
        "macro_specificity": float(metrics["specificity"].mean()),
        "scientific_role": "Public simulated paired text/structured proxy benchmark. Useful for extraction mechanics and prompt comparison; not evidence of performance on real RA EHR notes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public interim clinical-text benchmarks while restricted MIPA notes are unavailable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deepseek-sample-size", type=int, default=100)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audits = {}
    for name, fn in [
        ("nhse_synthetic_clinical_notes", audit_nhse),
        ("asclepius_synthetic_clinical_notes", audit_asclepius),
        ("pmc_patients", audit_pmc_patients),
        ("synthea", audit_synthea),
    ]:
        try:
            audits[name] = {"status": "COMPLETED", **fn()}
        except Exception as exc:  # noqa: BLE001
            audits[name] = {"status": "FAILED", "error": str(exc)}

    simsum = _load_simsum()
    regex_normal, normal_metrics = _evaluate_regex_simsum(simsum, "text")
    regex_advanced, advanced_metrics = _evaluate_regex_simsum(simsum, "advanced_text")
    normal_metrics.to_csv(output_dir / "simsum_regex_normal_metrics.csv", index=False)
    advanced_metrics.to_csv(output_dir / "simsum_regex_advanced_metrics.csv", index=False)
    deepseek = evaluate_deepseek_simsum(simsum, args.deepseek_sample_size, output_dir)

    summary = {
        "schema_version": "public-interim-benchmarks-v0.1",
        "purpose": "Interim public-data experiments while MIMIC/MIPA restricted notes access is pending.",
        "dataset_audits": audits,
        "simsum_proxy_benchmark": {
            "dataset_n": int(len(simsum)),
            "features": list(SIMSUM_FEATURES),
            "regex_normal_text": regex_normal,
            "regex_advanced_text": regex_advanced,
            "deepseek_advanced_text": deepseek,
            "interpretation": (
                "SimSUM supplies paired simulated clinical notes and structured ground truth, so real extraction metrics can be computed. "
                "It is respiratory-domain simulated data, not RA/MIMIC; these metrics validate mechanics, not C3 real-clinical generalisation."
            ),
        },
        "scientific_conclusion": {
            "can_continue_engineering_and_proxy_evaluation_while_waiting_for_mimic": True,
            "can_replace_authorised_mipa_for_c3": False,
            "can_replace_mimic_gold_text_structured_for_c4": False,
            "recommended_interim_order": [
                "SimSUM paired extraction benchmark",
                "NHSE longitudinal note/evidence stress test",
                "Synthea structured cohort mechanics",
                "Asclepius large-scale prompt robustness",
                "PMC-Patients real-literature domain-shift/open-ended discovery",
            ],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
