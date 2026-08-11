"""Scientific conclusion gates for RA comorbidity recovery studies.

This module separates public feasibility evidence, external published benchmarks, the project's
own clinical validation, confirmatory recovery, and downstream RA impact. It also provides
subject-cluster bootstrap intervals so repeated notes from the same patient are not treated as
independent observations.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ScientificThresholds:
    min_ra_positive_notes: int = 100
    min_ra_unique_subjects: int = 75
    min_fraction_ra_with_two_other_phenotypes: float = 0.70
    external_llm_macro_f1: float = 0.85
    min_code_missed_gold: int = 20
    min_text_positive_code_absent_ppv: float = 0.85
    min_bootstrap_replicates: int = 500


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int | None]:
    gold_positive = sum(int(row["gold"]) == 1 for row in rows)
    structured_tp = sum(int(row["gold"]) == 1 and int(row["structured"]) == 1 for row in rows)
    text_tp = sum(int(row["gold"]) == 1 and int(row["text"]) == 1 for row in rows)
    combined_tp = sum(
        int(row["gold"]) == 1 and (int(row["structured"]) == 1 or int(row["text"]) == 1)
        for row in rows
    )
    code_missed_gold = sum(int(row["gold"]) == 1 and int(row["structured"]) == 0 for row in rows)
    recovered = sum(
        int(row["gold"]) == 1 and int(row["text"]) == 1 and int(row["structured"]) == 0
        for row in rows
    )
    text_positive_code_absent = sum(
        int(row["text"]) == 1 and int(row["structured"]) == 0 for row in rows
    )
    structured_sensitivity = _ratio(structured_tp, gold_positive)
    text_sensitivity = _ratio(text_tp, gold_positive)
    combined_sensitivity = _ratio(combined_tp, gold_positive)
    return {
        "n_rows": len(rows),
        "gold_positive": gold_positive,
        "structured_sensitivity": structured_sensitivity,
        "text_sensitivity": text_sensitivity,
        "combined_sensitivity": combined_sensitivity,
        "combined_minus_structured_sensitivity": (
            combined_sensitivity - structured_sensitivity
            if combined_sensitivity is not None and structured_sensitivity is not None
            else None
        ),
        "gold_positive_code_absent": code_missed_gold,
        "gold_positive_text_positive_code_absent": recovered,
        "recoverable_code_missed_fraction": _ratio(recovered, code_missed_gold),
        "gold_ppv_among_text_positive_code_absent": _ratio(recovered, text_positive_code_absent),
    }


def subject_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260723,
) -> dict[str, object]:
    """Bootstrap patients, retaining all note/phenotype rows within each sampled subject cluster."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    by_subject: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        for required in ("subject_id", "gold", "text", "structured"):
            if required not in row:
                raise ValueError(f"Scientific bootstrap row missing {required!r}")
        by_subject[str(row["subject_id"])].append(row)
    subjects = sorted(by_subject)
    if not subjects:
        raise ValueError("No subjects available for scientific bootstrap")

    rng = random.Random(seed)
    deltas: list[float] = []
    recovery_rates: list[float] = []
    ppvs: list[float] = []
    for _ in range(n_bootstrap):
        sampled_rows: list[Mapping[str, object]] = []
        for _ in range(len(subjects)):
            sampled_rows.extend(by_subject[rng.choice(subjects)])
        metrics = _metrics(sampled_rows)
        delta = metrics["combined_minus_structured_sensitivity"]
        recovery = metrics["recoverable_code_missed_fraction"]
        ppv = metrics["gold_ppv_among_text_positive_code_absent"]
        if delta is not None:
            deltas.append(float(delta))
        if recovery is not None:
            recovery_rates.append(float(recovery))
        if ppv is not None:
            ppvs.append(float(ppv))

    def interval(values: Sequence[float]) -> dict[str, float | None]:
        return {
            "estimate_mean": mean(values) if values else None,
            "ci95_low": _quantile(values, 0.025),
            "ci95_high": _quantile(values, 0.975),
        }

    return {
        "method": "subject_cluster_bootstrap",
        "n_subjects": len(subjects),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "combined_minus_structured_sensitivity": interval(deltas),
        "recoverable_code_missed_fraction": interval(recovery_rates),
        "gold_ppv_among_text_positive_code_absent": interval(ppvs),
    }


def _stage12_state(summary: Mapping[str, object] | None) -> tuple[str | None, str, bool]:
    if summary is None:
        return None, "PENDING_REAL_DATA", False
    raw = str(dict(summary.get("acceptance", {})).get("final_status", "UNKNOWN"))
    if raw == "PASS":
        return raw, "PASS", True
    if raw.startswith("FAIL"):
        return raw, "NOT_SUPPORTED", True
    return raw, "PENDING_REAL_DATA", False


def _recovery_state(summary: Mapping[str, object] | None, stage12_status: str) -> tuple[str | None, str, bool]:
    if summary is None:
        if stage12_status == "NOT_SUPPORTED":
            return None, "BLOCKED_BY_STAGE12", False
        return None, "PENDING_REAL_DATA", False
    raw = str(summary.get("status", "UNKNOWN"))
    if raw == "SUPPORTED_RECOVERABLE_UNDERRECORDING":
        return raw, "PASS", True
    if raw in {"INCONCLUSIVE_OR_UNSUPPORTED_RECOVERY", "INSUFFICIENT_RECOVERY_OPPORTUNITIES"}:
        return raw, "NOT_SUPPORTED", True
    if raw == "NOT_CONFIRMATORY_ELIGIBLE":
        return raw, "BLOCKED_BY_UPSTREAM_GATES", False
    return raw, "PENDING_REAL_DATA", False


def assess_current_scientific_evidence(
    *,
    public_mipa_summary: Mapping[str, object],
    external_llm_macro_f1: float | None,
    split_protocol_summary: Mapping[str, object] | None = None,
    own_stage12_summary: Mapping[str, object] | None = None,
    own_discordance_summary: Mapping[str, object] | None = None,
    own_recovery_summary: Mapping[str, object] | None = None,
    bsrbr_impact_summary: Mapping[str, object] | None = None,
    thresholds: ScientificThresholds = ScientificThresholds(),
) -> dict[str, object]:
    """Build a conservative, claim-specific scientific conclusion matrix.

    A research stage can be scientifically complete with a negative/null result. External literature
    is context only and never acts as a required gate for MedCode's own scientific conclusions.
    """
    ra_subset = dict(public_mipa_summary.get("ra_subset", {}))
    ra_subject_audit = dict(ra_subset.get("subject_id_audit", {}))
    cooccurrence = dict(ra_subset.get("cooccurrence_summary", {}))
    at_least_two = dict(cooccurrence.get("at_least_2_other_phenotypes", {}))

    n_ra_notes = int(ra_subset.get("n_ra_positive_notes", 0) or 0)
    n_ra_subjects = int(ra_subject_audit.get("n_unique", 0) or 0)
    fraction_two = float(at_least_two.get("fraction", 0.0) or 0.0)
    feasibility_pass = (
        n_ra_notes >= thresholds.min_ra_positive_notes
        and n_ra_subjects >= thresholds.min_ra_unique_subjects
        and fraction_two >= thresholds.min_fraction_ra_with_two_other_phenotypes
    )

    protocol_pass = False
    protocol_basis: dict[str, object] = {}
    if split_protocol_summary is not None:
        benchmark_protocol = dict(split_protocol_summary.get("benchmark_comparison_protocol", {}))
        ra_protocol = dict(split_protocol_summary.get("ra_confirmatory_protocol", {}))
        ra_protocol_audit = dict(ra_protocol.get("audit", {}))
        reporting_rule = dict(split_protocol_summary.get("reporting_rule", {}))
        protocol_pass = (
            benchmark_protocol.get("any_subject_overlap_between_validation_and_test") is True
            and ra_protocol_audit.get("subject_disjoint") is True
            and reporting_rule.get("do_not_mix_protocol_metrics") is True
        )
        protocol_basis = {
            "official_admission_split_has_patient_overlap": benchmark_protocol.get("any_subject_overlap_between_validation_and_test"),
            "ra_confirmatory_subject_disjoint": ra_protocol_audit.get("subject_disjoint"),
            "protocol_metrics_must_not_be_mixed": reporting_rule.get("do_not_mix_protocol_metrics"),
        }

    external_support = (
        external_llm_macro_f1 is not None
        and float(external_llm_macro_f1) >= thresholds.external_llm_macro_f1
    )

    own_stage12_raw, own_stage12_status, stage12_terminal = _stage12_state(own_stage12_summary)
    recovery_raw, recovery_status, recovery_terminal = _recovery_state(own_recovery_summary, own_stage12_status)

    discordance_status = None
    if own_discordance_summary is not None:
        discordance_status = str(
            dict(own_discordance_summary.get("interpretation", {})).get("status", "UNKNOWN")
        )

    bsrbr_complete = False
    bsrbr_effect = None
    if bsrbr_impact_summary is not None:
        bsrbr_complete = bool(bsrbr_impact_summary.get("analysis_complete", False))
        bsrbr_effect = bsrbr_impact_summary.get("effect_conclusion")
    bsrbr_status = "PASS_ASSESSED" if bsrbr_complete else "PENDING_BSRBR_RA"

    conclusions = [
        {
            "id": "C0_PROTOCOL_INTEGRITY",
            "claim": "Benchmark-comparison and patient-level confirmatory evaluation protocols are explicitly separated and audited for subject leakage.",
            "status": "PASS" if protocol_pass else "PENDING_METHOD_AUDIT",
            "stage_complete": protocol_pass,
            "gating": True,
            "basis": protocol_basis,
            "allowed_language": "Report official/protocol-matched benchmark results separately from subject-disjoint confirmatory RA results.",
        },
        {
            "id": "C1_MIPA_RA_PILOT_FEASIBILITY",
            "claim": "MIPA contains enough RA-positive benchmark observations and co-occurring labelled phenotypes for a controlled RA comorbidity NLP pilot.",
            "status": "PASS" if feasibility_pass else "NOT_SUPPORTED",
            "stage_complete": True,
            "gating": True,
            "basis": {
                "n_ra_positive_notes": n_ra_notes,
                "n_ra_unique_subjects": n_ra_subjects,
                "fraction_ra_notes_with_at_least_2_other_phenotypes": fraction_two,
            },
            "allowed_language": "Suitable proof-of-concept benchmark for RA comorbidity phenotyping; not an RA prevalence cohort.",
        },
        {
            "id": "C2_EXTERNAL_LLM_PHENOTYPING_FEASIBILITY",
            "claim": "High-performing LLM phenotyping is feasible on MIPA under published study settings.",
            "status": "EXTERNAL_SUPPORT_ONLY" if external_support else "NOT_SUPPORTED",
            "stage_complete": True,
            "gating": False,
            "basis": {"published_external_macro_f1": external_llm_macro_f1},
            "allowed_language": "Published evidence supports feasibility only; do not use it as MedCode performance or as a direct comparator unless the evaluation protocol is matched.",
        },
        {
            "id": "C3_OWN_EVIDENCE_GROUNDED_MODEL_PERFORMANCE",
            "claim": "The MedCode evidence-grounded local pipeline meets the pre-specified Stage 1/2 performance and evidence gates on authorised MIPA notes.",
            "status": own_stage12_status,
            "stage_complete": stage12_terminal,
            "gating": True,
            "basis": {"own_stage12_final_status": own_stage12_raw},
            "allowed_language": "Report MedCode clinical performance only after an authorised real-note run reaches a terminal Stage 1/2 result.",
        },
        {
            "id": "C4_RECOVERABLE_STRUCTURED_UNDERRECORDING",
            "claim": "Validated free-text phenotyping recovers clinician-supported comorbidities absent from the matched structured phenotype definition.",
            "status": recovery_status,
            "stage_complete": recovery_terminal,
            "gating": True,
            "basis": {
                "discordance_eligibility_status": discordance_status,
                "confirmatory_recovery_status": recovery_raw,
            },
            "allowed_language": "A supported, null, or negative result is scientifically valid; only SUPPORTED_RECOVERABLE_UNDERRECORDING supports the directional recovery claim.",
        },
        {
            "id": "C5_RA_MULTIMORBIDITY_IMPACT_ASSESSED",
            "claim": "The effect of text-enhanced ascertainment on patient-level RA multimorbidity estimates, clusters, trajectories, or outcomes has been evaluated in appropriate longitudinal/representative RA data.",
            "status": bsrbr_status,
            "stage_complete": bsrbr_complete,
            "gating": True,
            "basis": {"effect_conclusion": bsrbr_effect},
            "allowed_language": "A material change, a small change, or a null effect are all valid outcomes once the prespecified BSRBR-RA analysis is complete.",
        },
    ]

    # Decision path: a valid negative result is not a software/scientific-process failure.
    if own_stage12_status == "NOT_SUPPORTED":
        overall_status = "SCIENTIFIC_NO_GO_STAGE12"
        scientifically_determined = True
        next_gate = "STOP_OR_REDESIGN_PHENOTYPING_METHOD"
    elif recovery_terminal and recovery_status == "NOT_SUPPORTED":
        overall_status = "SCIENTIFIC_CONCLUSION_DETERMINED_NO_RECOVERY_SUPPORT"
        scientifically_determined = True
        next_gate = "DO_NOT_CLAIM_RECOVERABLE_UNDERRECORDING"
    elif own_stage12_status == "PASS" and recovery_status == "PASS" and bsrbr_complete:
        overall_status = "SCIENTIFIC_PROGRAM_COMPLETE"
        scientifically_determined = True
        next_gate = "REPORT_PRESPECIFIED_RESULTS"
    elif own_stage12_status == "PASS" and recovery_status == "PASS":
        overall_status = "GO_BSRBR_RA"
        scientifically_determined = False
        next_gate = "BSRBR_RA_IMPACT_ANALYSIS"
    elif own_stage12_status == "PASS":
        overall_status = "GO_STAGE3_CONFIRMATORY_RECOVERY"
        scientifically_determined = False
        next_gate = "GOLD_TEXT_STRUCTURED_CONFIRMATORY_ANALYSIS"
    else:
        overall_status = "NOT_YET_SCIENTIFICALLY_DETERMINED"
        scientifically_determined = False
        next_gate = "AUTHORISED_REAL_NOTE_STAGE12"

    required_method_gates_ready = protocol_pass and feasibility_pass
    return {
        "schema_version": "scientific-acceptance-v0.3.2",
        "thresholds": thresholds.__dict__,
        "conclusions": conclusions,
        "overall": {
            "scientifically_determined": scientifically_determined,
            "status": overall_status,
            "next_gate": next_gate,
            "required_method_gates_ready": required_method_gates_ready,
            "all_directional_claims_supported": (
                own_stage12_status == "PASS" and recovery_status == "PASS"
            ),
            "reason": (
                "External benchmark success is non-gating. A terminal negative/null empirical result is scientifically valid. "
                "Positive MedCode claims require the project's own authorised data and prespecified confirmatory gates."
            ),
        },
    }


def assess_confirmatory_recovery(
    *,
    rows: Sequence[Mapping[str, object]],
    stage12_final_status: str,
    structured_scope_validated: bool,
    complete_pair_coverage: bool,
    thresholds: ScientificThresholds = ScientificThresholds(),
    n_bootstrap: int = 2000,
    seed: int = 20260723,
) -> dict[str, object]:
    """Apply confirmatory gates to a subject-aware Gold×Text×Structured result table."""
    point = _metrics(rows)
    bootstrap = subject_cluster_bootstrap(rows, n_bootstrap=n_bootstrap, seed=seed)
    eligible = (
        stage12_final_status == "PASS"
        and structured_scope_validated
        and complete_pair_coverage
        and n_bootstrap >= thresholds.min_bootstrap_replicates
    )
    code_missed = int(point["gold_positive_code_absent"] or 0)
    ppv = point["gold_ppv_among_text_positive_code_absent"]
    delta_ci_low = dict(bootstrap["combined_minus_structured_sensitivity"]).get("ci95_low")

    recovery_opportunity = code_missed >= thresholds.min_code_missed_gold
    ppv_pass = ppv is not None and float(ppv) >= thresholds.min_text_positive_code_absent_ppv
    improvement_supported = delta_ci_low is not None and float(delta_ci_low) > 0.0

    if not eligible:
        status = "NOT_CONFIRMATORY_ELIGIBLE"
    elif not recovery_opportunity:
        status = "INSUFFICIENT_RECOVERY_OPPORTUNITIES"
    elif ppv_pass and improvement_supported:
        status = "SUPPORTED_RECOVERABLE_UNDERRECORDING"
    else:
        status = "INCONCLUSIVE_OR_UNSUPPORTED_RECOVERY"

    return {
        "schema_version": "confirmatory-recovery-v0.3.2",
        "point_estimates": point,
        "subject_cluster_bootstrap": bootstrap,
        "gates": {
            "stage12_pass": stage12_final_status == "PASS",
            "structured_scope_validated": structured_scope_validated,
            "complete_pair_coverage": complete_pair_coverage,
            "minimum_bootstrap_replicates": n_bootstrap >= thresholds.min_bootstrap_replicates,
            "minimum_code_missed_gold": recovery_opportunity,
            "text_positive_code_absent_ppv": ppv_pass,
            "combined_sensitivity_improvement_ci95_low_gt_zero": improvement_supported,
        },
        "status": status,
        "interpretation": {
            "negative_result_is_scientifically_valid": True,
            "no_effect_does_not_equal_pipeline_failure": True,
            "unit_of_resampling": "subject_id",
        },
    }
