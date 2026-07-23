# MedCode v0.2.1 selective-policy result

| Policy | TEST coverage | TEST Accuracy@1 | Release eligible |
|---|---:|---:|---:|
| previous_reused_model_selection_validation | 65.2% | 84.0% | False |
| independent_empirical_policy_calibration | 61.0% | 86.9% | False |
| conservative_lower_bound_policy | 56.0% | 91.1% | True |

AUTO_CANDIDATE may be enabled only when a threshold selected on the disjoint policy-calibration set has a one-sided binomial accuracy lower bound at or above the prespecified target. Otherwise AUTO coverage is zero and cases remain human-routed.
