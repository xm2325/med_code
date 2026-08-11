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


def _split_summary():
    return {
        "benchmark_comparison_protocol": {
            "any_subject_overlap_between_validation_and_test": True,
        },
        "ra_confirmatory_protocol": {
            "audit": {"subject_disjoint": True},
        },
        "reporting_rule": {"do_not_mix_protocol_metrics": True},
    }


def test_current_evidence_passes_method_feasibility_but_remains_pending_real_data():
    result = assess_current_scientific_evidence(
        public_mipa_summary=_public_summary(),
        split_protocol_summary=_split_summary(),
        external_llm_macro_f1=0.891,
    )
    by_id = {row["id"]: row for row in result["conclusions"]}
    assert by_id["C0_PROTOCOL_INTEGRITY"]["status"] == "PASS"
    assert by_id["C1_MIPA_RA_PILOT_FEASIBILITY"]["status"] == "PASS"
    assert by_id["C2_EXTERNAL_LLM_PHENOTYPING_FEASIBILITY"]["status"] == "EXTERNAL_SUPPORT_ONLY"
    assert by_id["C2_EXTERNAL_LLM_PHENOTYPING_FEASIBILITY"]["gating"] is False
    assert by_id["C3_OWN_EVIDENCE_GROUNDED_MODEL_PERFORMANCE"]["status"] == "PENDING_REAL_DATA"
    assert by_id["C4_RECOVERABLE_STRUCTURED_UNDERRECORDING"]["status"] == "PENDING_REAL_DATA"
    assert by_id["C5_RA_MULTIMORBIDITY_IMPACT_ASSESSED"]["status"] == "PENDING_BSRBR_RA"
    assert result["overall"]["status"] == "NOT_YET_SCIENTIFICALLY_DETERMINED"
    assert result["overall"]["scientifically_determined"] is False
    assert result["overall"]["required_method_gates_ready"] is True


def test_external_benchmark_cannot_substitute_for_own_stage12_pass():
    result = assess_current_scientific_evidence(
        public_mipa_summary=_public_summary(),
        split_protocol_summary=_split_summary(),
        external_llm_macro_f1=0.95,
        own_stage12_summary={"acceptance": {"final_status": "PASS_AUTOMATED_PENDING_HUMAN_EVIDENCE_AUDIT"}},
    )
    by_id = {row["id"]: row for row in result["conclusions"]}
    assert by_id["C2_EXTERNAL_LLM_PHENOTYPING_FEASIBILITY"]["status"] == "EXTERNAL_SUPPORT_ONLY"
    assert by_id["C3_OWN_EVIDENCE_GROUNDED_MODEL_PERFORMANCE"]["status"] == "PENDING_REAL_DATA"
    assert result["overall"]["next_gate"] == "AUTHORISED_REAL_NOTE_STAGE12"


def test_terminal_stage12_failure_is_valid_scientific_no_go_not_pending_forever():
    result = assess_current_scientific_evidence(
        public_mipa_summary=_public_summary(),
        split_protocol_summary=_split_summary(),
        external_llm_macro_f1=0.891,
        own_stage12_summary={"acceptance": {"final_status": "FAIL_AUTOMATED_GATE"}},
    )
    by_id = {row["id"]: row for row in result["conclusions"]}
    assert by_id["C3_OWN_EVIDENCE_GROUNDED_MODEL_PERFORMANCE"]["status"] == "NOT_SUPPORTED"
    assert by_id["C3_OWN_EVIDENCE_GROUNDED_MODEL_PERFORMANCE"]["stage_complete"] is True
    assert by_id["C4_RECOVERABLE_STRUCTURED_UNDERRECORDING"]["status"] == "BLOCKED_BY_STAGE12"
    assert result["overall"]["status"] == "SCIENTIFIC_NO_GO_STAGE12"
    assert result["overall"]["scientifically_determined"] is True


def _supported_rows():
    rows = []
    for subject in range(40):
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


def test_negative_recovery_is_scientifically_terminal_not_relabelled_positive():
    rows = []
    for subject in range(40):
        structured = 1 if subject < 20 else 0
        rows.append(
            {
                "subject_id": f"S{subject}",
                "gold": 1,
                "text": structured,
                "structured": structured,
            }
        )
    recovery = assess_confirmatory_recovery(
        rows=rows,
        stage12_final_status="PASS",
        structured_scope_validated=True,
        complete_pair_coverage=True,
        n_bootstrap=500,
        seed=11,
    )
    assert recovery["status"] == "INCONCLUSIVE_OR_UNSUPPORTED_RECOVERY"
    assert recovery["interpretation"]["negative_result_is_scientifically_valid"] is True

    matrix = assess_current_scientific_evidence(
        public_mipa_summary=_public_summary(),
        split_protocol_summary=_split_summary(),
        external_llm_macro_f1=0.891,
        own_stage12_summary={"acceptance": {"final_status": "PASS"}},
        own_recovery_summary=recovery,
    )
    by_id = {row["id"]: row for row in matrix["conclusions"]}
    assert by_id["C4_RECOVERABLE_STRUCTURED_UNDERRECORDING"]["status"] == "NOT_SUPPORTED"
    assert by_id["C4_RECOVERABLE_STRUCTURED_UNDERRECORDING"]["stage_complete"] is True
    assert matrix["overall"]["status"] == "SCIENTIFIC_CONCLUSION_DETERMINED_NO_RECOVERY_SUPPORT"
    assert matrix["overall"]["scientifically_determined"] is True


def test_bsrbr_null_effect_can_complete_program_after_supported_recovery():
    recovery = {"status": "SUPPORTED_RECOVERABLE_UNDERRECORDING"}
    result = assess_current_scientific_evidence(
        public_mipa_summary=_public_summary(),
        split_protocol_summary=_split_summary(),
        external_llm_macro_f1=0.891,
        own_stage12_summary={"acceptance": {"final_status": "PASS"}},
        own_recovery_summary=recovery,
        bsrbr_impact_summary={"analysis_complete": True, "effect_conclusion": "null_or_small_effect"},
    )
    by_id = {row["id"]: row for row in result["conclusions"]}
    assert by_id["C5_RA_MULTIMORBIDITY_IMPACT_ASSESSED"]["status"] == "PASS_ASSESSED"
    assert result["overall"]["status"] == "SCIENTIFIC_PROGRAM_COMPLETE"
    assert result["overall"]["scientifically_determined"] is True
