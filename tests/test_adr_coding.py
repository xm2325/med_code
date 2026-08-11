from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from cohortcoder.adr_coding import DEMO_ADR_TERMS, OneLineADRCoder


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_rows() -> list[dict]:
    return json.loads((ROOT / "examples" / "ra_adr_synthetic_50.json").read_text(encoding="utf-8"))


def test_all_50_synthetic_records_map_to_expected_meddra_code_with_verbatim_evidence():
    rows = _synthetic_rows()
    assert len(rows) == 50
    assert len({row["record_id"] for row in rows}) == 50

    coder = OneLineADRCoder()
    for row in rows:
        result = coder.map_line(row["text"], record_id=row["record_id"])
        assert result["primary"] is not None, row["record_id"]
        assert result["primary"]["code"] == row["expected_code"], row["record_id"]
        assert result["primary"]["term"] == row["expected_term"], row["record_id"]
        evidence = result["primary"]["evidence"]
        assert row["text"][evidence["start"] : evidence["end"]] == evidence["quote"]
        assert row["drug"] in result["drug_mentions"]
        assert result["synthetic_demo_only"] is True


def test_uncertain_records_route_to_top_k_human_choice():
    rows = {row["record_id"]: row for row in _synthetic_rows()}
    coder = OneLineADRCoder()
    for record_id in ("SYN-031", "SYN-046", "SYN-050"):
        result = coder.map_line(rows[record_id]["text"])
        assert result["route"] == "TOP_K_HUMAN_CHOICE"
        assert result["primary"]["assertion"] == "uncertain"
        assert result["confidence"] <= 0.74


def test_negated_event_is_excluded_when_an_affirmed_event_exists_later_in_line():
    text = "No anaphylaxis during infliximab infusion; the RA patient was later admitted with pneumonia."
    result = OneLineADRCoder().map_line(text)
    assert result["primary"]["code"] == "10035664"
    assert result["primary"]["assertion"] == "affirmed"
    excluded = [candidate for candidate in result["candidates"] if candidate["code"] == "10002218"]
    assert excluded and excluded[0]["assertion"] == "negated"


def test_chinese_event_and_drug_are_supported():
    result = OneLineADRCoder().map_line("类风湿复诊：甲氨蝶呤后出现恶心。")
    assert result["primary"]["code"] == "10028813"
    assert result["drug_mentions"] == ["Methotrexate"]
    assert result["route"] == "AUTO_CANDIDATE"


def test_multiple_affirmed_events_do_not_auto_route():
    result = OneLineADRCoder().map_line("RA patient had nausea and vomiting after methotrexate.")
    assert result["route"] == "TOP_K_HUMAN_CHOICE"
    assert {item["code"] for item in result["encoded_events"]} == {"10028813", "10047700"}


def test_no_demo_term_routes_to_full_review_without_fabricating_a_code():
    result = OneLineADRCoder().map_line("RA patient felt generally unwell after methotrexate.")
    assert result["route"] == "FULL_EXPERT_REVIEW"
    assert result["primary"] is None
    assert result["encoded_events"] == []


def test_empty_line_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        OneLineADRCoder().map_line("   ")


def test_demo_terminology_codes_are_unique_eight_digit_values():
    codes = [term.code for term in DEMO_ADR_TERMS]
    assert len(codes) == len(set(codes))
    assert all(code.isdigit() and len(code) == 8 for code in codes)


def test_standalone_html_embeds_all_50_records_and_record_picker():
    html = (ROOT / "demo" / "ra_adr_meddra_demo.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="medcodePayload">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert len(embedded["records"]) == 50
    assert {row["record_id"] for row in embedded["records"]} == {
        f"SYN-{index:03d}" for index in range(1, 51)
    }
    assert 'id="sampleSelect"' in html
    assert 'id="randomBtn"' in html
    assert 'data-method="baseline"' in html
    assert 'data-method="prompt"' in html
    assert 'data-method="rag"' in html
    assert embedded["method"]["baseline_executed_in_demo"] is True
    assert embedded["method"]["prompt_only_executed_in_demo"] is False
    assert embedded["method"]["rag_executed_in_demo"] is False
    assert "fetch(" not in html
    assert "<script src=" not in html


def test_english_standalone_html_is_self_contained_and_translated():
    html = (ROOT / "demo" / "ra_adr_meddra_demo.en.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="medcodePayload">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert len(embedded["records"]) == 50
    assert '<html lang="en">' in html
    assert not re.search(r"[\u3400-\u9fff]", html)
    assert 'data-method="baseline"' in html
    assert 'data-method="prompt"' in html
    assert 'data-method="rag"' in html
    assert "fetch(" not in html
    assert "<script src=" not in html
