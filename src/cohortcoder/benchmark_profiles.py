from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    task_type: str
    coding_system: str
    split_unit: str
    gold_structure: str
    primary_metrics: tuple[str, ...]
    rationale_unit: str
    access_note: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primary_metrics"] = list(self.primary_metrics)
        return payload


CADEC_MEDDRA = BenchmarkProfile(
    name="cadec_meddra_normalization",
    task_type="single_label_concept_normalization",
    coding_system="MedDRA",
    split_unit="source_document",
    gold_structure="one MedDRA concept per annotated mention",
    primary_metrics=("accuracy_at_1", "accuracy_at_5", "candidate_recall_at_10"),
    rationale_unit="annotated mention / supporting source span",
    access_note="Public research corpus subject to the source data licence; full licensed MedDRA distributions are not redistributed.",
)


MIMIC_IV_ICD10 = BenchmarkProfile(
    name="mimic_iv_note_icd10_multilabel",
    task_type="multilabel_document_coding",
    coding_system="ICD-10",
    split_unit="subject_id",
    gold_structure="multiple ICD-10 diagnosis codes per hospitalization/discharge summary",
    primary_metrics=("micro_f1", "macro_f1", "precision_at_10", "recall_at_10"),
    rationale_unit="record-and-code evidence span(s)",
    access_note="Credentialed PhysioNet data. Do not commit notes, derived note text, or patient-level annotations to a public repository.",
)


PROFILES = {
    CADEC_MEDDRA.name: CADEC_MEDDRA,
    MIMIC_IV_ICD10.name: MIMIC_IV_ICD10,
}


def get_benchmark_profile(name: str) -> BenchmarkProfile:
    try:
        return PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark profile: {name}. Available: {sorted(PROFILES)}") from exc
