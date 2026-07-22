"""MedCode research prototype."""

from .advanced import AdvancedModelConfig, AdvancedSingleLabelCoder, CrossEncoderCandidateReranker, DenseSemanticIndex
from .analysis import annotate_prediction_diagnostics, choose_threshold_max_coverage, coverage_accuracy_curve, failure_summary, policy_stress_test, subgroup_metrics, write_evaluation_plots
from .benchmark_profiles import CADEC_MEDDRA, MIMIC_IV_ICD10, BenchmarkProfile, get_benchmark_profile
from .cadec import audit_cadec_records, parse_cadec, write_cadec_audit_artifacts
from .candidate_rationales import build_candidate_rationales, build_review_packet
from .core import HistoricalCoder, accuracy_at_k, coverage_accuracy
from .explain import EvidenceSpan, build_explanation_record, explain_predictions, extract_evidence_spans, write_explanation_artifacts
from .explanation_quality import apply_explanation_quality_gate, evaluate_explanation_quality, summarize_explanation_quality
from .feedback import ExpertFeedback, append_feedback_jsonl, feedback_summary, feedback_to_training_memory, hash_reviewer_id, validate_feedback
from .knowledge import load_terminology_knowledge, prepare_terminology_knowledge
from .llm import DeepSeekRationaleClient, ExternalLLMPolicyError, validate_llm_rationale
from .llm_rerank import DeepSeekCandidateReranker, validate_rerank_payload
from .mimic_audit import audit_mimic_records, write_mimic_audit_artifacts
from .multilabel import MultiLabelHistoricalCoder, ranking_metrics, threshold_metrics
from .rationale_metrics import evaluate_rationale_overlap, validate_rationale_offsets
from .results import ResultsContract, build_results_contract, contract_from_benchmark_metadata, write_results_contract
from .uncertainty import ReviewRoutingPolicy, candidate_uncertainty, simple_ood_flag

__version__ = "0.0.15"

__all__ = [
    "HistoricalCoder", "AdvancedModelConfig", "AdvancedSingleLabelCoder", "DenseSemanticIndex", "CrossEncoderCandidateReranker",
    "MultiLabelHistoricalCoder", "accuracy_at_k", "coverage_accuracy", "ranking_metrics", "threshold_metrics",
    "ResultsContract", "build_results_contract", "contract_from_benchmark_metadata", "write_results_contract",
    "annotate_prediction_diagnostics", "choose_threshold_max_coverage", "coverage_accuracy_curve", "failure_summary", "policy_stress_test", "subgroup_metrics", "write_evaluation_plots",
    "EvidenceSpan", "extract_evidence_spans", "build_explanation_record", "explain_predictions", "write_explanation_artifacts",
    "apply_explanation_quality_gate", "evaluate_explanation_quality", "summarize_explanation_quality",
    "load_terminology_knowledge", "prepare_terminology_knowledge", "DeepSeekRationaleClient", "DeepSeekCandidateReranker",
    "ExternalLLMPolicyError", "validate_llm_rationale", "validate_rerank_payload",
    "BenchmarkProfile", "CADEC_MEDDRA", "MIMIC_IV_ICD10", "get_benchmark_profile", "evaluate_rationale_overlap", "validate_rationale_offsets",
    "parse_cadec", "audit_cadec_records", "write_cadec_audit_artifacts", "audit_mimic_records", "write_mimic_audit_artifacts",
    "ReviewRoutingPolicy", "candidate_uncertainty", "simple_ood_flag", "build_candidate_rationales", "build_review_packet",
    "ExpertFeedback", "append_feedback_jsonl", "feedback_summary", "feedback_to_training_memory", "hash_reviewer_id", "validate_feedback",
]
