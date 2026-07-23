import pandas as pd

from cohortcoder.ra_comorbidity import confidence_review_curve, evaluate_discordance, public_mipa_ra_summary


def test_hidden_comorbidity_recovery_is_separate_from_code_recall():
    df = pd.DataFrame([
        {"subject_id": "1", "phenotype": "vte_past", "gold": 1, "code": 0, "text": 1, "confidence": 0.95},
        {"subject_id": "2", "phenotype": "vte_past", "gold": 1, "code": 0, "text": 0, "confidence": 0.20},
        {"subject_id": "3", "phenotype": "vte_past", "gold": 1, "code": 1, "text": 1, "confidence": 0.90},
        {"subject_id": "4", "phenotype": "vte_past", "gold": 0, "code": 0, "text": 1, "confidence": 0.55},
    ])
    metrics, patterns = evaluate_discordance(df)
    row = metrics.loc[metrics["phenotype"] == "vte_past"].iloc[0]
    assert row["gold_positive_code_negative"] == 2
    assert row["hidden_recovered_by_text"] == 1
    assert row["hcrr"] == 0.5
    assert row["code_recall"] == 1 / 3
    assert "G1_C0_T1" in set(patterns["pattern"])


def test_review_curve_is_proposal_level():
    df = pd.DataFrame([
        {"subject_id": "1", "phenotype": "x", "gold": 1, "code": 0, "text": 1, "confidence": 0.95},
        {"subject_id": "2", "phenotype": "x", "gold": 0, "code": 0, "text": 1, "confidence": 0.60},
    ])
    curve = confidence_review_curve(df, thresholds=[0.5, 0.9])
    at_09 = curve.loc[curve["threshold"] == 0.9].iloc[0]
    assert at_09["sent_to_review"] == 1
    assert at_09["auto_precision"] == 1.0


def test_public_mipa_summary_deduplicates_patients():
    labels = pd.DataFrame([
        {"note_id": "a", "subject_id": 1, "hadm_id": 10, "hypertension": 1, "depression": 0, "none": 0, "rheumatoid_arthritis": 1},
        {"note_id": "b", "subject_id": 1, "hadm_id": 11, "hypertension": 1, "depression": 1, "none": 0, "rheumatoid_arthritis": 1},
        {"note_id": "c", "subject_id": 2, "hadm_id": 12, "hypertension": 0, "depression": 0, "none": 0, "rheumatoid_arthritis": 1},
    ])
    summary, counts = public_mipa_ra_summary(labels)
    assert summary["ra_positive_admissions"] == 3
    assert summary["unique_ra_patients"] == 2
    assert summary["population_prevalence_claim_allowed"] is False
    assert counts.loc["hypertension", "positive_patients"] == 1
    assert counts.loc["depression", "positive_patients"] == 1
