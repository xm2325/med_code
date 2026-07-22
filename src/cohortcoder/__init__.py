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
from .results import ResultsContract, build_results_contract, contract_from_benchmark_metadata, write_results_contract

__version__ = "0.0.9"

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
]
