"""MedCode research prototype."""

from .analysis import (
    annotate_prediction_diagnostics,
    choose_threshold_max_coverage,
    coverage_accuracy_curve,
    failure_summary,
    policy_stress_test,
    subgroup_metrics,
    write_evaluation_plots,
)
from .core import HistoricalCoder, accuracy_at_k, coverage_accuracy
from .explain import (
    EvidenceSpan,
    build_explanation_record,
    explain_predictions,
    extract_evidence_spans,
    write_explanation_artifacts,
)
from .knowledge import load_terminology_knowledge, prepare_terminology_knowledge
from .llm import DeepSeekRationaleClient, ExternalLLMPolicyError, validate_llm_rationale
from .results import ResultsContract, build_results_contract, contract_from_benchmark_metadata, write_results_contract

__version__ = "0.0.10"

__all__ = [
    "HistoricalCoder",
    "accuracy_at_k",
    "coverage_accuracy",
    "ResultsContract",
    "build_results_contract",
    "contract_from_benchmark_metadata",
    "write_results_contract",
    "annotate_prediction_diagnostics",
    "choose_threshold_max_coverage",
    "coverage_accuracy_curve",
    "failure_summary",
    "policy_stress_test",
    "subgroup_metrics",
    "write_evaluation_plots",
    "EvidenceSpan",
    "extract_evidence_spans",
    "build_explanation_record",
    "explain_predictions",
    "write_explanation_artifacts",
    "load_terminology_knowledge",
    "prepare_terminology_knowledge",
    "DeepSeekRationaleClient",
    "ExternalLLMPolicyError",
    "validate_llm_rationale",
]
