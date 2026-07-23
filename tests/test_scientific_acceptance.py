from __future__ import annotations

from cohortcoder.scientific_acceptance import (
    assess_confirmatory_recovery,
    assess_current_scientific_evidence,
    subject_cluster_bootstrap,
)


def _public_summary():
    return {
        "ra_subset": {
            "n_ra_positive_notes": 164,
            "subject_id_audit": {"n_unique": 99},
            "cooccurrence_summary": {
                "at_least_2_other_phenotypes": {"n": 121, "fraction": 121 / 164},
            },
        }
    }


def test_current_evidence_passes_feasibility_but_not_overall_scientific_confirmation():
    result = assess_current_scientific_evidence(
        public_mipa_summary=_public_summary(),
        external_llm_macro_f1=0.891,
    )
    by_id = {row["id"]: row for row in result["conclusions"]}
    assert by_id["C1_MIPA_RA_PILOT_FEASIBILITY"]["status"] == "PASS"
    assert by_id["C2_LLM_PHENOTYPING_CAPABILITY"]["status"] == "EXTERNAL_SUPPORT_ONLY"
    assert by_id["C3_OWN_EVIDENCE_GROUNDED_MODEL_PERFORMANCE"]["status"] == "PENDING_REAL_DATA"
    assert by_id["C4_RECOVERABLE_STRUCTURED_UNDERRECORDING"]["status"] == "PENDING_REAL_DATA"
    assert by_id["C5_RA_MULTIMORBIDITY_IMPACT"]["status"] == "PENDING_BSRBR_RA"
    assert result["overall"]["status"] == "NOT_YET_SCIENTIFICALLY_CONFIRMED"
    assert result["overall"]["scientifically_confirmed"] is False


def test_external_benchmark_cannot_substitute_for_own_stage12_pass():
    result = assess_current_scientific_evidence(
        public_mipa_summary=_public_summary(),
        external_llm_macro_f1=0.95,
        own_stage12_summary={"acceptance": {"final_status": "PASS_AUTOMATED_PENDING_HUMAN_EVIDENCE_AUDIT"}},
    )
    by_id = {row["id"]: row for row in result["conclusions"]}
    assert by_id["C2_LLM_PHENOTYPING_CAPABILITY"]["status"] == "EXTERNAL_SUPPORT_ONLY"
    assert by_id["C3_OWN_EVIDENCE_GROUNDED_MODEL_PERFORMANCE"]["status"] == "PENDING_REAL_DATA"


def _supported_rows():
    rows = []
    for subject in range(40):
        # Two rows per patient emulate repeated observations; structured coding misses the second half.
        structured = 1 if subject < 20 else 0
        for note in range(2):
            rows.append(
                {
                    "subject_id": f"S{subject}",
                    "gold": 1,
                    "text": 1,
                    "structured": structured,
                    "note_id": f"S{subject}-N{note}",
                }
            )
    return rows


def test_subject_cluster_bootstrap_uses_unique_subjects_and_detects_improvement():
    result = subject_cluster_bootstrap(_supported_rows(), n_bootstrap=500, seed=7)
    assert result["n_subjects"] == 40
    interval = result["combined_minus_structured_sensitivity"]
    assert interval["ci95_low"] is not None
    assert interval["ci95_low"] > 0


def test_confirmatory_recovery_supported_only_after_all_upstream_gates_pass():
    supported = assess_confirmatory_recovery(
        rows=_supported_rows(),
        stage12_final_status="PASS",
        structured_scope_validated=True,
        complete_pair_coverage=True,
        n_bootstrap=500,
        seed=7,
    )
    assert supported["status"] == "SUPPORTED_RECOVERABLE_UNDERRECORDING"
    assert supported["gates"]["combined_sensitivity_improvement_ci95_low_gt_zero"] is True

    blocked = assess_confirmatory_recovery(
        rows=_supported_rows(),
        stage12_final_status="PASS_AUTOMATED_PENDING_HUMAN_EVIDENCE_AUDIT",
        structured_scope_validated=True,
        complete_pair_coverage=True,
        n_bootstrap=500,
        seed=7,
    )
    assert blocked["status"] == "NOT_CONFIRMATORY_ELIGIBLE"


def test_negative_or_inconclusive_recovery_is_not_relabelled_as_success():
    rows = []
    for subject in range(40):
        structured = 1 if subject < 20 else 0
        # Text fails to recover structured misses.
        text = structured
        rows.append(
            {
                "subject_id": f"S{subject}",
                "gold": 1,
                "text": text,
                "structured": structured,
            }
        )
    result = assess_confirmatory_recovery(
        rows=rows,
        stage12_final_status="PASS",
        structured_scope_validated=True,
        complete_pair_coverage=True,
        n_bootstrap=500,
        seed=11,
    )
    assert result["status"] == "INCONCLUSIVE_OR_UNSUPPORTED_RECOVERY"
    assert result["interpretation"]["negative_result_is_scientifically_valid"] is True
