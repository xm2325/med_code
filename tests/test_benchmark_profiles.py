from cohortcoder.benchmark_profiles import CADEC_MEDDRA, MIMIC_IV_ICD10, get_benchmark_profile


def test_cadec_and_mimic_are_explicitly_different_tasks():
    assert CADEC_MEDDRA.task_type == "single_label_concept_normalization"
    assert MIMIC_IV_ICD10.task_type == "multilabel_document_coding"
    assert CADEC_MEDDRA.split_unit == "source_document"
    assert MIMIC_IV_ICD10.split_unit == "subject_id"
    assert get_benchmark_profile(MIMIC_IV_ICD10.name).coding_system == "ICD-10"
