import pandas as pd

from cohortcoder.data import load_coding_records


def test_missing_mention_stays_blank_for_document_level_coding(tmp_path):
    path = tmp_path / "records.csv"
    pd.DataFrame([
        {"record_id": "r1", "text": "Long clinical narrative with a diagnosis later in the note."},
    ]).to_csv(path, index=False)
    records = load_coding_records(path)
    assert records.iloc[0].mention == ""
    assert records.iloc[0].text.startswith("Long clinical narrative")
