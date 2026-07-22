# v0.1.1 real MedNorm + DeepSeek result status

The evaluation implementation is present, but **no real DeepSeek accuracy is reported in this directory yet**.

The dedicated GitHub Actions run was attempted and retried. In both attempts the `real-evaluation` job terminated before any workflow step was recorded (`steps = null`, `logs_url = null`). Therefore the job did not provide evidence that checkout, dependency installation, MedNorm download, secret access, or a DeepSeek API request occurred.

It would be misleading to fill this directory with invented metrics.

When the runner executes successfully, `.github/workflows/deepseek-real-mednorm.yml` will produce and publish:

- deterministic real MedNorm/CADEC-derived baseline Accuracy@1/@3/@5;
- paired baseline vs DeepSeek reranked Accuracy@1 on a fixed-seed real-data subset;
- fixed candidate Recall@5;
- seen/unseen-code statistics;
- validation-selected AUTO / TOP_K_HUMAN_CHOICE / FULL_EXPERT_REVIEW routing;
- an explicitly labelled oracle Top-K human-choice upper bound;
- real case examples where **every candidate option** has the original phrase as verbatim evidence and a validated candidate-specific rationale;
- complete per-case artifacts as an Actions artifact.

The evaluation is a TRAIN-derived closed-code diagnostic unless an authorised full MedDRA terminology resource is supplied separately. It must not be described as a full open-set MedDRA benchmark.
