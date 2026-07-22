import json

import pandas as pd

from cohortcoder.cadec import audit_cadec_records, parse_cadec
from cohortcoder.source_evidence import apply_task_input_spans


def _make_cadec(tmp_path):
    root = tmp_path / "cadec"
    for name in ["text", "original", "meddra"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    text = "I had pain in both legs."
    (root / "text" / "r1.txt").write_text(text, encoding="utf-8")
    (root / "original" / "r1.ann").write_text(
        "T1\tADE 6 10\tpain\nT2\tADE 6 10;19 23\tpain legs\n",
        encoding="utf-8",
    )
    (root / "meddra" / "r1.ann").write_text(
        "N1\tReference T1 MedDRA:10000001\nN2\tReference T2 MedDRA:10000002\n",
        encoding="utf-8",
    )
    return root, text


def test_cadec_parser_preserves_discontinuous_spans(tmp_path):
    root, _ = _make_cadec(tmp_path)
    df, stats = parse_cadec(root)
    assert len(df) == 2
    discontinuous = df[df["annotation_id"] == "T2"].iloc[0]
    assert json.loads(discontinuous["spans_json"]) == [[6, 10], [19, 23]]
    assert bool(discontinuous["is_discontinuous"])
    assert bool(discontinuous["offset_match"])
    assert stats["offset_match_rate"] == 1.0


def test_cadec_audit_parses_string_booleans_correctly():
    records = pd.DataFrame([
        {"record_id": "r1", "annotation_id": "T1", "text": "pain", "mention": "pain", "gold_code": "10000001", "offset_match": "False", "is_discontinuous": "False"},
    ])
    terminology = pd.DataFrame([{"code": "10000001", "term": "Pain"}])
    audit = audit_cadec_records(records, terminology)
    assert audit["offset_match_rate"] == 0.0
    assert "offset_match_rate_below_threshold" in audit["hard_failures"]


def test_task_input_spans_override_ambiguous_string_search():
    text = "pain earlier, then later pain"
    explanations = [{
        "record_id": "r1",
        "text": text,
        "predicted_code": "10000001",
        "predicted_term": "Pain",
        "evidence_spans": [],
        "evidence_quotes": [],
        "evidence_verbatim": False,
    }]
    source = pd.DataFrame([{"spans_json": json.dumps([[25, 29]]), "text": text}])
    out = apply_task_input_spans(explanations, source)
    assert out[0]["evidence_spans"][0]["start"] == 25
    assert out[0]["evidence_spans"][0]["quote"] == "pain"
    assert out[0]["evidence_spans"][0]["source"] == "task_input_span"
