"""MedCode explainable clinical coding toolkit."""

from .advanced import AdvancedModelConfig, AdvancedSingleLabelCoder, CrossEncoderCandidateReranker, DenseSemanticIndex
from .analysis import annotate_prediction_diagnostics, choose_threshold_max_coverage, coverage_accuracy_curve, failure_summary, policy_stress_test, subgroup_metrics, write_evaluation_plots
from .audit_replay import ARTIFACT_SCHEMA_VERSION, build_audit_bundle, decision_trace, sha256_file, validate_audit_bundle
from .benchmark_profiles import CADEC_MEDDRA, MIMIC_IV_ICD10, BenchmarkProfile, get_benchmark_profile
from .cadec import audit_cadec_records, parse_cadec, write_cadec_audit_artifacts
from .candidate_generation import AliasAwareHybridCoder, CandidateGenerationConfig
from .candidate_rationales import build_candidate_rationales, build_review_packet
from .core import HistoricalCoder, accuracy_at_k, coverage_accuracy
from .deepseek_real_eval import DeepSeekRealCandidateEvaluator, validate_multi_candidate_payload
from .discordance import evaluate_three_way_discordance, write_discordance_outputs
from .explain import EvidenceSpan, build_explanation_record, explain_predictions, extract_evidence_spans, write_explanation_artifacts
from .explanation_quality import apply_explanation_quality_gate, evaluate_explanation_quality, summarize_explanation_quality
from .feedback import ExpertFeedback, append_feedback_jsonl, feedback_summary, feedback_to_training_memory, hash_reviewer_id, validate_feedback
from .knowledge import load_terminology_knowledge, prepare_terminology_knowledge
from .llm import DeepSeekRationaleClient, ExternalLLMPolicyError, validate_llm_rationale
from .llm_rerank import DeepSeekCandidateReranker, validate_rerank_payload
from .mednorm import OFFICIAL_MEDNORM, assign_cross_dataset_split, build_train_derived_terminology, fetch_hf_mirror_dataframe, fetch_hf_mirror_rows, mednorm_data_card, prepare_mednorm_single_meddra
from .mimic_audit import audit_mimic_records, write_mimic_audit_artifacts
from .mipa_local_inference import InferenceConfig, build_prompt_payload, generate_local_predictions, parse_model_response
from .mipa_phenotyping import AcceptanceThresholds, DEFAULT_PHENOTYPES, evaluate_mipa_predictions, write_evaluation_outputs
from .mipa_strict_acceptance import EvidenceAuditCoveragePolicy, evaluate_mipa_predictions_strict
from .multilabel import MultiLabelHistoricalCoder, ranking_metrics, threshold_metrics
from .rationale_metrics import evaluate_rationale_overlap, validate_rationale_offsets
from .registry import register_experiment, stable_json_hash, write_data_card, write_model_card
from .release import V010_REQUIRED_CAPABILITIES, build_release_manifest, validate_release_manifest, write_release_manifest
from .release_gate import evaluate_release_readiness
from .results import ResultsContract, build_results_contract, contract_from_benchmark_metadata, write_results_contract
from .review_service import ReviewQueue
from .scientific_acceptance import ScientificThresholds, assess_confirmatory_recovery, assess_current_scientific_evidence, subject_cluster_bootstrap
from .selective_policy import SelectivePolicySelection, apply_frozen_threshold, one_sided_binomial_lower_bound, select_threshold_by_accuracy_lower_bound
from .uncertainty import ReviewRoutingPolicy, candidate_uncertainty, simple_ood_flag

__version__ = "0.3.1"
__all__ = [name for name in globals() if not name.startswith("_")]
