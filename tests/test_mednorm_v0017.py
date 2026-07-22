import pandas as pd

from cohortcoder.mednorm import assign_cross_dataset_split, build_train_derived_terminology, prepare_mednorm_single_meddra


def test_mednorm_adapter_and_train_only_candidate_space():
    raw = pd.DataFrame([
        {"original_dataset":"X","instance_id":"1","phrase":"muscle pain","single_mapped_meddra_codes":"1001"},
        {"original_dataset":"X","instance_id":"2","phrase":"aching muscles","single_mapped_meddra_codes":"1001"},
        {"original_dataset":"CADEC","instance_id":"3","phrase":"novel symptom","single_mapped_meddra_codes":"9999"},
    ])
    records = prepare_mednorm_single_meddra(raw)
    split = assign_cross_dataset_split(records, test_source="CADEC", val_fraction=0.5)
    train = split[split.split == "train"]
    terminology = build_train_derived_terminology(train)
    assert "9999" not in set(terminology.code.astype(str))
    assert set(split[split.source_dataset == "CADEC"].split) == {"test"}
