from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_DISCORDANCE_COLUMNS = {"subject_id", "phenotype", "gold", "code", "text"}
DEFAULT_ID_COLUMNS = ("note_id", "subject_id", "hadm_id")


@dataclass(frozen=True)
class HiddenComorbiditySummary:
    n_pairs: int
    gold_positive: int
    gold_positive_code_negative: int
    hidden_recovered_by_text: int
    hcrr: float | None
    text_only_candidates: int
    text_only_ppv_against_gold: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "n_pairs": self.n_pairs,
            "gold_positive": self.gold_positive,
            "gold_positive_code_negative": self.gold_positive_code_negative,
            "hidden_recovered_by_text": self.hidden_recovered_by_text,
            "hcrr": self.hcrr,
            "text_only_candidates": self.text_only_candidates,
            "text_only_ppv_against_gold": self.text_only_ppv_against_gold,
        }


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int | None]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def validate_discordance_table(df: pd.DataFrame) -> None:
    missing = REQUIRED_DISCORDANCE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for column in ("gold", "code", "text"):
        invalid = ~df[column].isin([0, 1])
        if invalid.any():
            raise ValueError(f"{column} must contain only binary 0/1 values")


def hidden_comorbidity_summary(df: pd.DataFrame) -> HiddenComorbiditySummary:
    validate_discordance_table(df)
    hidden_gold = (df["gold"] == 1) & (df["code"] == 0)
    recovered = hidden_gold & (df["text"] == 1)
    text_only = (df["text"] == 1) & (df["code"] == 0)
    text_only_true = text_only & (df["gold"] == 1)
    return HiddenComorbiditySummary(
        n_pairs=int(len(df)),
        gold_positive=int((df["gold"] == 1).sum()),
        gold_positive_code_negative=int(hidden_gold.sum()),
        hidden_recovered_by_text=int(recovered.sum()),
        hcrr=_safe_div(int(recovered.sum()), int(hidden_gold.sum())),
        text_only_candidates=int(text_only.sum()),
        text_only_ppv_against_gold=_safe_div(int(text_only_true.sum()), int(text_only.sum())),
    )


def evaluate_discordance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Gold x structured-Code x Text discordance.

    `code` is deliberately not named `gold_code`: structured coding is a comparator,
    not automatically a clinical reference standard.
    """
    validate_discordance_table(df)
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("__OVERALL__", df)]
    groups.extend((str(name), group) for name, group in df.groupby("phenotype", sort=True))
    for phenotype, group in groups:
        code_metrics = binary_metrics(group["gold"], group["code"])
        text_metrics = binary_metrics(group["gold"], group["text"])
        union = ((group["code"] == 1) | (group["text"] == 1)).astype(int)
        union_metrics = binary_metrics(group["gold"], union)
        hidden = hidden_comorbidity_summary(group)
        rows.append({
            "phenotype": phenotype,
            **hidden.to_dict(),
            "code_precision": code_metrics["precision"],
            "code_recall": code_metrics["recall"],
            "code_f1": code_metrics["f1"],
            "text_precision": text_metrics["precision"],
            "text_recall": text_metrics["recall"],
            "text_f1": text_metrics["f1"],
            "code_or_text_recall": union_metrics["recall"],
        })
    patterns = (
        df.assign(pattern=("G" + df["gold"].astype(str) + "_C" + df["code"].astype(str) + "_T" + df["text"].astype(str)))
        .groupby(["phenotype", "pattern"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    return pd.DataFrame(rows), patterns


def confidence_review_curve(df: pd.DataFrame, *, thresholds: Iterable[float] = np.arange(0.50, 0.951, 0.05)) -> pd.DataFrame:
    """Evaluate a proposal-level selective policy for text-positive candidates."""
    validate_discordance_table(df)
    if "confidence" not in df.columns:
        return pd.DataFrame()
    rows = []
    predicted_positive = int((df["text"] == 1).sum())
    for threshold in thresholds:
        auto = ((df["text"] == 1) & (df["confidence"] >= float(threshold))).astype(int)
        metrics = binary_metrics(df["gold"], auto)
        sent_to_review = int(((df["text"] == 1) & (df["confidence"] < float(threshold))).sum())
        rows.append({
            "threshold": round(float(threshold), 2),
            "auto_precision": metrics["precision"],
            "auto_recall": metrics["recall"],
            "predicted_positive_candidates": predicted_positive,
            "sent_to_review": sent_to_review,
            "review_fraction_of_text_positive": _safe_div(sent_to_review, predicted_positive),
        })
    return pd.DataFrame(rows)


def public_mipa_ra_summary(labels: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarise public MIPA labels for an RA feasibility benchmark.

    The output de-duplicates repeated admissions at patient level. MIPA is ICD-enriched,
    so these counts must not be interpreted as population comorbidity prevalence.
    """
    required = {"subject_id", "rheumatoid_arthritis"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"MIPA labels missing required columns: {sorted(missing)}")
    phenotype_cols = [c for c in labels.columns if c not in set(DEFAULT_ID_COLUMNS) | {"none", "rheumatoid_arthritis"}]
    ra_admissions = labels[labels["rheumatoid_arthritis"] == 1].copy()
    patient = ra_admissions.groupby("subject_id", as_index=True)[phenotype_cols].max()
    patient["n_other_phenotypes"] = patient[phenotype_cols].sum(axis=1)
    counts = patient[phenotype_cols].sum().sort_values(ascending=False).rename("positive_patients").to_frame()
    counts["pct_of_unique_ra_patients_in_mipa"] = 100.0 * counts["positive_patients"] / len(patient)
    summary: dict[str, object] = {
        "mipa_admissions": int(len(labels)),
        "ra_positive_admissions": int(len(ra_admissions)),
        "unique_ra_patients": int(ra_admissions["subject_id"].nunique()),
        "ra_patients_with_at_least_1_other_phenotype": int((patient["n_other_phenotypes"] >= 1).sum()),
        "ra_patients_with_at_least_2_other_phenotypes": int((patient["n_other_phenotypes"] >= 2).sum()),
        "ra_patients_with_at_least_3_other_phenotypes": int((patient["n_other_phenotypes"] >= 3).sum()),
        "median_other_phenotypes_per_patient": float(patient["n_other_phenotypes"].median()),
        "max_other_phenotypes_per_patient": int(patient["n_other_phenotypes"].max()),
        "population_prevalence_claim_allowed": False,
        "reason": "MIPA candidate notes were enriched using phenotype-related ICD signals before multilabel expert annotation.",
    }
    return summary, counts
