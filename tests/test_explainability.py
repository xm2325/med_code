import json

import pandas as pd
import pytest

from cohortcoder.core import HistoricalCoder
from cohortcoder.explain import build_explanation_record, extract_evidence_spans
from cohortcoder.knowledge import prepare_terminology_knowledge
from cohortcoder.llm import DeepSeekRationaleClient, ExternalLLMPolicyError, validate_llm_rationale


def test_evidence_spans_are_verbatim_with_offsets():
    text = "At follow-up the patient reported severe muscle pain in both thighs. No rash was reported."
    spans = extract_evidence_spans(text, mention="severe muscle pain", term="Myalgia", synonyms="muscle pain|aching muscles")
    assert spans
    assert spans[0].quote == "severe muscle pain"
    assert text[spans[0].start:spans[0].end] == spans[0].quote


def test_knowledge_loader_keeps_clean_term_and_builds_search_text():
    terminology = pd.DataFrame([{
        "system": "MedDRA",
        "code": "100",
        "term": "Myalgia",
        "synonyms": "muscle pain|aching muscles",
        "definition": "Pain affecting muscle tissue",
    }])
    out = prepare_terminology_knowledge(terminology)
    assert out.iloc[0].term == "Myalgia"
    assert "muscle pain" in out.iloc[0].search_text
    assert "Pain affecting muscle tissue" in out.iloc[0].search_text


def test_one_run_rejects_mixed_coding_systems():
    terminology = pd.DataFrame([
        {"system": "ICD-10", "code": "A", "term": "Alpha"},
        {"system": "MedDRA", "code": "B", "term": "Beta"},
    ])
    with pytest.raises(ValueError):
        prepare_terminology_knowledge(terminology)


def test_grounded_explanation_has_faithfulness_scores():
    history = pd.DataFrame([
        {"text": "aching muscles", "mention": "aching muscles", "gold_code": "A", "gold_term": "Myalgia"},
        {"text": "weak legs", "mention": "weak legs", "gold_code": "B", "gold_term": "Muscular weakness"},
    ])
    terminology = prepare_terminology_knowledge(pd.DataFrame([
        {"system": "DEMO", "code": "A", "term": "Myalgia", "synonyms": "muscle pain|aching muscles"},
        {"system": "DEMO", "code": "B", "term": "Muscular weakness", "synonyms": "weak muscles|weak legs"},
    ]))
    coder = HistoricalCoder(history_weight=0.25).fit(history, terminology)
    pred = coder.predict_one("severe muscle pain")
    row = {
        "record_id": "r1",
        "text": "The patient described severe muscle pain in both thighs.",
        "mention": "severe muscle pain",
        "predicted_code": pred.code,
        "predicted_term": pred.term,
        "confidence": pred.confidence,
        "decision": "HUMAN_REVIEW",
        "historical_cases_json": json.dumps(pred.historical_cases),
    }
    explanation = build_explanation_record(row, terminology, coder=coder)
    assert explanation["predicted_code"] == "A"
    assert explanation["evidence_verbatim"] is True
    assert "severe muscle pain" in explanation["evidence_quotes"]
    assert explanation["faithfulness"]["sufficiency_gap"] is not None
    assert explanation["faithfulness"]["comprehensiveness_drop"] is not None


def test_llm_validator_rejects_code_change_and_hallucinated_quote():
    valid, errors = validate_llm_rationale(
        {"code": "B", "rationale": "Because...", "evidence_quotes": ["invented evidence"]},
        locked_code="A",
        allowed_quotes=["severe muscle pain"],
    )
    assert valid is False
    assert "code_changed" in errors
    assert "non_verbatim_or_unapproved_evidence" in errors


def test_external_llm_is_blocked_for_restricted_data():
    with pytest.raises(ExternalLLMPolicyError):
        DeepSeekRationaleClient._check_data_policy(allow_external_llm=True, data_classification="restricted")
