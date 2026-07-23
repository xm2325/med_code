# MedCode v0.2 end-to-end observed result

> Public MedNorm / CADEC-held-out, TRAIN-derived closed-code diagnostic.

| Metric | Result |
|---|---:|
| Retrieval Accuracy@1 (500 TEST) | 67.0% |
| Retrieval Recall@5 | 84.4% |
| Retrieval Recall@20 | 88.6% |
| Retrieval Recall@50 | 90.0% |
| DeepSeek paired subset | 50 |
| DeepSeek valid response rate | 98.0% |
| Paired retrieval Accuracy@1 | 56.0% |
| Pipeline Accuracy@1 with DeepSeek/fallback | 62.0% |
| Pipeline delta | 6.0% |
| Fixed candidate Recall@5 | 78.0% |
| Cases corrected by DeepSeek | 5 |
| Cases harmed by DeepSeek | 2 |

Every displayed candidate retains the original real phrase as evidence and a candidate-specific rationale. This is not a licensed full-MedDRA open-set evaluation.
