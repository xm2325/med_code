from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import re
from typing import Any

import pandas as pd

from .analysis import (
    annotate_prediction_diagnostics,
    choose_threshold_max_coverage,
    coverage_accuracy_curve,
    failure_summary,
    policy_stress_test,
    subgroup_metrics,
)
from .core import HistoricalCoder, accuracy_at_k
from .results import contract_from_benchmark_metadata, write_results_contract


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_records(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    if not {"record_id", "text"}.issubset(df.columns):
        raise ValueError("records require record_id and text")
    if "mention" not in df:
        df["mention"] = df["text"]
    if "gold_code" not in df:
        df["gold_code"] = ""
    if "gold_term" not in df:
        df["gold_term"] = ""
    for col in ["record_id", "text", "mention", "gold_code", "gold_term"]:
        df[col] = df[col].astype(str)
    return df


def load_terminology(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    if not {"code", "term"}.issubset(df.columns):
        raise ValueError("terminology requires code,term columns")
    if "synonyms" in df:
        df["term"] = df["term"].astype(str) + " " + df["synonyms"].astype(str).str.replace("|", " ", regex=False)
    return df[["code", "term"]].astype(str).drop_duplicates("code").reset_index(drop=True)


def assign_document_splits(df: pd.DataFrame, seed: int = 42, train: float = 0.70, val: float = 0.15) -> pd.DataFrame:
    out = df.copy()
    mapping = {}
    for record_id in out["record_id"].astype(str).unique():
        u = int.from_bytes(sha256(f"{seed}:{record_id}".encode()).digest()[:8], "big") / 2**64
        mapping[record_id] = "train" if u < train else ("val" if u < train + val else "test")
    out["split"] = out["record_id"].astype(str).map(mapping)
    return out


def assert_document_disjoint(df: pd.DataFrame) -> None:
    if "split" not in df:
        raise ValueError("split column missing")
    memberships = df.groupby("record_id")["split"].nunique()
    if (memberships > 1).any():
        raise ValueError("document leakage: record_id occurs in multiple splits")


def parse_cadec(cadec_root: str | Path, output_csv: str | Path) -> pd.DataFrame:
    root = Path(cadec_root)
    if (root / "cadec").exists():
        root = root / "cadec"
    text_dir, original_dir, meddra_dir = root / "text", root / "original", root / "meddra"
    if not all(path.exists() for path in [text_dir, original_dir, meddra_dir]):
        raise FileNotFoundError("Expected CADEC text/, original/, and meddra/ directories")

    rows = []
    stats = {"documents": 0, "normalizations": 0, "matched": 0, "missing_target": 0}
    for meddra_path in sorted(meddra_dir.glob("*.ann")):
        stem = meddra_path.stem
        text_path = text_dir / f"{stem}.txt"
        original_path = original_dir / f"{stem}.ann"
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        lines = []
        if original_path.exists():
            lines += original_path.read_text(encoding="utf-8", errors="replace").splitlines()
        meddra_lines = meddra_path.read_text(encoding="utf-8", errors="replace").splitlines()
        lines += meddra_lines
        bounds = {}
        for line in lines:
            if not re.match(r"^T\d+\t", line):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            meta = parts[1].split(" ", 1)
            if len(meta) < 2:
                continue
            spans = []
            for segment in meta[1].split(";"):
                nums = re.findall(r"\d+", segment)
                if len(nums) >= 2:
                    spans.append((int(nums[0]), int(nums[1])))
            if not spans:
                continue
            bounds[parts[0]] = {
                "mention": parts[2],
                "entity_type": meta[0],
                "start": min(a for a, _ in spans),
                "end": max(b for _, b in spans),
            }
        stats["documents"] += 1
        for line in meddra_lines:
            target = re.search(r"\b(T\d+)\b", line)
            code = re.search(r"(?:MedDRA\s*[:#]?\s*)?(\d{6,9})\b", line, re.I)
            if not (target and code):
                continue
            stats["normalizations"] += 1
            bound = bounds.get(target.group(1))
            if not bound:
                stats["missing_target"] += 1
                continue
            stats["matched"] += 1
            rows.append({
                "record_id": stem,
                "text": text,
                "mention": bound["mention"],
                "entity_type": bound["entity_type"],
                "start": bound["start"],
                "end": bound["end"],
                "gold_code": code.group(1),
                "gold_term": "",
                "source_dataset": "CADEC",
            })

    df = pd.DataFrame(rows).drop_duplicates(["record_id", "mention", "gold_code"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No CADEC MedDRA normalizations parsed")
    stats.update({
        "rows": len(df),
        "unique_documents": int(df.record_id.nunique()),
        "unique_codes": int(df.gold_code.nunique()),
        "match_rate": stats["matched"] / stats["normalizations"] if stats["normalizations"] else 0.0,
    })
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    _dump(output.with_suffix(".parse_stats.json"), stats)
    return df


def _predict_frame(coder: HistoricalCoder, df: pd.DataFrame) -> pd.DataFrame:
    query = df["mention"].where(df["mention"].str.len() > 0, df["text"])
    predictions = coder.predict(query)
    rows = []
    for (_, record), prediction in zip(df.iterrows(), predictions):
        rows.append({
            "record_id": record.record_id,
            "gold_code": record.gold_code,
            "gold_term": record.gold_term,
            "predicted_code": prediction.code,
            "predicted_term": prediction.term,
            "confidence": prediction.confidence,
            "correct": int(prediction.code == str(record.gold_code)),
            "candidates_json": json.dumps(prediction.candidates),
            "historical_cases_json": json.dumps(prediction.historical_cases),
        })
    return pd.DataFrame(rows)


def _metric_for_subgroup(table: pd.DataFrame, subgroup: str, metric: str) -> float | None:
    row = table[table["subgroup"] == subgroup]
    if row.empty or pd.isna(row.iloc[0][metric]):
        return None
    return float(row.iloc[0][metric])


def _write_html_report(
    output: Path,
    contract_status: str,
    contract_reportable: bool,
    metrics: dict[str, Any],
    open_set: pd.DataFrame,
    policy_stress: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    html = f"""<html><body>
<h1>MedCode v0.0.9 benchmark</h1>
<p>Status: <b>{contract_status}</b></p>
<p>Reportable: <b>{contract_reportable}</b></p>
<h2>Primary held-out TEST metrics</h2>
<ul>
<li>Accuracy@1: {metrics['accuracy_at_1']:.3f}</li>
<li>Accuracy@5: {metrics['accuracy_at_5']:.3f}</li>
<li>AUTO candidate rate: {metrics['auto_candidate_rate']:.3f}</li>
<li>AUTO candidate accuracy: {metrics['auto_candidate_accuracy']}</li>
<li>Human review rate: {metrics['human_review_rate']:.3f}</li>
<li>Selected historical-memory weight: {metrics['selected_history_weight']:.2f}</li>
<li>Historical-memory Acc@1 delta vs terminology-only: {metrics['historical_memory_accuracy_delta']:.3f}</li>
</ul>
<h2>Seen/unseen code analysis</h2>
{open_set.to_html(index=False)}
<h2>Validation-selected policy stress test</h2>
<p>Each threshold below is selected on validation only, then evaluated unchanged on TEST.</p>
{policy_stress.to_html(index=False)}
<h2>Failure taxonomy</h2>
{failures.to_html(index=False)}
<p><b>Note:</b> the TEST coverage-accuracy curve is descriptive evaluation, not a source of deployment threshold selection.</p>
</body></html>"""
    (output / "report.html").write_text(html, encoding="utf-8")


def run_real_benchmark(
    records: pd.DataFrame,
    terminology: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_auto_accuracy: float = 0.95,
    external_human_reference: bool = True,
    data_is_synthetic: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = records.copy()
    if "split" not in data or not {"train", "val", "test"}.issubset(set(data["split"].astype(str))):
        data = assign_document_splits(data, seed=seed)
    assert_document_disjoint(data)
    train = data[data.split == "train"].copy()
    val = data[data.split == "val"].copy()
    test = data[data.split == "test"].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("train/val/test must all be non-empty")

    tuning = []
    for history_weight in [0.0, 0.25, 0.50, 0.75, 1.0]:
        coder = HistoricalCoder(history_weight=history_weight, top_k=10).fit(train, terminology)
        validation_predictions = _predict_frame(coder, val)
        candidate_lists = [json.loads(value) for value in validation_predictions.candidates_json]
        tuning.append({
            "history_weight": history_weight,
            "val_accuracy_at_1": float(validation_predictions.correct.mean()),
            "val_accuracy_at_5": accuracy_at_k(validation_predictions.gold_code, candidate_lists, 5),
        })
    tuning_df = pd.DataFrame(tuning)
    tuning_df.to_csv(output / "model_selection.csv", index=False)
    best = tuning_df.sort_values(
        ["val_accuracy_at_1", "val_accuracy_at_5", "history_weight"],
        ascending=[False, False, True],
    ).iloc[0]

    selected_coder = HistoricalCoder(history_weight=float(best.history_weight), top_k=10).fit(train, terminology)
    baseline_coder = HistoricalCoder(history_weight=0.0, top_k=10).fit(train, terminology)
    validation_predictions = _predict_frame(selected_coder, val)
    test_predictions = _predict_frame(selected_coder, test)
    baseline_test_predictions = _predict_frame(baseline_coder, test)

    # v0.0.9: choose the threshold that maximises validation coverage while meeting
    # the prespecified accuracy target. TEST remains untouched during policy selection.
    threshold = choose_threshold_max_coverage(validation_predictions, target_auto_accuracy)
    test_predictions["decision"] = "HUMAN_REVIEW"
    if threshold is not None:
        test_predictions.loc[test_predictions.confidence >= threshold, "decision"] = "AUTO_CANDIDATE"

    seen_codes = {str(code) for code in train.gold_code if str(code)}
    diagnostics = annotate_prediction_diagnostics(test_predictions, seen_codes, candidate_k=10)
    auto = diagnostics.decision == "AUTO_CANDIDATE"
    test_candidate_lists = [json.loads(value) for value in diagnostics.candidates_json]
    open_set = subgroup_metrics(diagnostics)
    coverage_curve = coverage_accuracy_curve(diagnostics)
    policy_stress = policy_stress_test(validation_predictions, diagnostics)
    failures = failure_summary(diagnostics)

    baseline_acc1 = float(baseline_test_predictions.correct.mean())
    selected_acc1 = float(diagnostics.correct.mean())
    historical_memory_value = {
        "terminology_only_test_accuracy_at_1": baseline_acc1,
        "selected_method_test_accuracy_at_1": selected_acc1,
        "selected_history_weight": float(best.history_weight),
        "accuracy_at_1_delta": selected_acc1 - baseline_acc1,
        "interpretation": "Descriptive held-out comparison; model selection used validation only.",
    }

    metrics = {
        "n_test": len(diagnostics),
        "accuracy_at_1": selected_acc1,
        "accuracy_at_5": accuracy_at_k(diagnostics.gold_code, test_candidate_lists, 5),
        "selected_history_weight": float(best.history_weight),
        "selected_threshold": threshold,
        "threshold_selection_rule": "max_validation_coverage_subject_to_target_accuracy",
        "auto_candidate_rate": float(auto.mean()),
        "auto_candidate_accuracy": float(diagnostics.loc[auto, "correct"].mean()) if auto.any() else None,
        "human_review_rate": float((~auto).mean()),
        "target_auto_accuracy": target_auto_accuracy,
        "seen_code_accuracy_at_1": _metric_for_subgroup(open_set, "seen_code", "accuracy_at_1"),
        "unseen_code_accuracy_at_1": _metric_for_subgroup(open_set, "unseen_code", "accuracy_at_1"),
        "candidate_recall_at_10": _metric_for_subgroup(open_set, "all", "candidate_recall_at_10"),
        "historical_memory_accuracy_delta": selected_acc1 - baseline_acc1,
    }

    overlap = set(train.record_id) & set(test.record_id)
    contract = contract_from_benchmark_metadata(
        external_human_reference=external_human_reference,
        group_disjoint_test=not overlap,
        candidate_dictionary_source="external",
        test_used_for_selection_or_tuning=False,
        data_is_synthetic=data_is_synthetic,
        provenance_recorded=True,
    )
    metrics["results_status"] = contract.status
    metrics["results_reportable"] = contract.reportable

    diagnostics.to_csv(output / "predictions.csv", index=False)
    validation_predictions.to_csv(output / "validation_predictions.csv", index=False)
    baseline_test_predictions.to_csv(output / "terminology_only_test_predictions.csv", index=False)
    open_set.to_csv(output / "open_set_metrics.csv", index=False)
    coverage_curve.to_csv(output / "coverage_accuracy.csv", index=False)
    policy_stress.to_csv(output / "policy_stress_test.csv", index=False)
    failures.to_csv(output / "failure_summary.csv", index=False)
    diagnostics[[
        "record_id", "gold_code", "predicted_code", "gold_candidate_rank",
        "code_novelty", "error_type", "confidence", "decision",
    ]].to_csv(output / "candidate_retrieval_diagnostics.csv", index=False)

    _dump(output / "metrics.json", metrics)
    _dump(output / "historical_memory_value.json", historical_memory_value)
    write_results_contract(output / "results_contract.json", contract)
    _dump(output / "leakage_audit.json", {"record_id_overlap": len(overlap)})
    _dump(output / "frozen_policy.json", {
        "version": "0.0.9",
        "history_weight": float(best.history_weight),
        "decision_threshold": threshold,
        "threshold_selection_rule": "max_validation_coverage_subject_to_target_accuracy",
        "target_auto_accuracy": target_auto_accuracy,
    })
    _dump(output / "experiment_manifest.json", {
        "version": "0.0.9",
        "seed": seed,
        "external_human_reference": external_human_reference,
        "data_is_synthetic": data_is_synthetic,
        "policy_targets": [0.90, 0.95, 0.98, 0.99],
    })
    _dump(output / "data_fingerprints.json", {
        "records_sha256": sha256(data.to_csv(index=False).encode()).hexdigest(),
        "terminology_sha256": sha256(terminology.to_csv(index=False).encode()).hexdigest(),
    })
    diagnostics[diagnostics.correct == 0].to_csv(output / "error_analysis.csv", index=False)
    _write_html_report(output, contract.status, contract.reportable, metrics, open_set, policy_stress, failures)
    return metrics


def predict_uncoded(
    historical: pd.DataFrame,
    terminology: pd.DataFrame,
    new_records: pd.DataFrame,
    frozen_policy: dict[str, Any],
) -> pd.DataFrame:
    coder = HistoricalCoder(history_weight=float(frozen_policy["history_weight"]), top_k=10).fit(historical, terminology)
    query = new_records["mention"].where(new_records["mention"].astype(str).str.len() > 0, new_records["text"])
    predictions = coder.predict(query)
    threshold = frozen_policy.get("decision_threshold")
    rows = []
    for (_, record), prediction in zip(new_records.iterrows(), predictions):
        decision = "AUTO_CANDIDATE" if threshold is not None and prediction.confidence >= float(threshold) else "HUMAN_REVIEW"
        rows.append({
            "record_id": record.record_id,
            "predicted_code": prediction.code,
            "predicted_term": prediction.term,
            "confidence": prediction.confidence,
            "decision": decision,
            "candidates_json": json.dumps(prediction.candidates),
            "historical_cases_json": json.dumps(prediction.historical_cases),
        })
    return pd.DataFrame(rows)
