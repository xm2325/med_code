# MedCode v0.0.13 — Model quality protocol

v0.0.13 adds stronger candidate retrieval and reranking without weakening the evidence or evaluation boundary.

## Single-label MedDRA stack

```text
clinical mention
    |
    +--> lexical terminology retrieval
    |
    +--> historical expert-coding memory
    |
    +--> optional biomedical dense retrieval
    |        user-supplied sentence-transformer model/path
    |
    v
frozen candidate pool
    |
    +--> optional cross-encoder reranker
    |
    +--> optional DeepSeek frozen-candidate reranker
    |
    v
selected code
    |
    v
exact evidence -> clinical context audit -> why-this-code rationale
    |
    v
explanation quality gate
    |
    +--> keep AUTO_CANDIDATE when all policy conditions pass
    |
    +--> downgrade to HUMAN_REVIEW when evidence quality fails
```

## Validation-only staged selection

The default advanced benchmark does not assume a larger model is better.

1. Tune historical-memory weight using validation only.
2. When a dense model is explicitly supplied, compare dense fusion weights using validation only.
3. When a cross-encoder is explicitly supplied, compare reranker fusion weights using validation only.
4. Break ties in favour of the simpler model.
5. Freeze the final configuration and evaluate TEST once.

The selected model family, weights, model names/paths and AUTO/REVIEW threshold are stored in `frozen_policy.json`.

## Dense model boundary

The repository does not bundle or silently download biomedical model weights. The user supplies a sentence-transformer-compatible model name or local path with `--dense-model`.

A dense backend contributes two semantic signals:

- query -> terminology concept similarity;
- query -> historical expert-coded example similarity, aggregated by code.

The dense signal is fused with the pre-existing clean lexical baseline. `history_weight=0` remains a terminology-only lexical ablation.

## Cross-encoder boundary

A cross-encoder is only allowed to rerank candidates already generated from the authorised terminology dictionary. It cannot add a code outside the candidate dictionary.

Cross-encoder use is optional and selected on validation. If it does not improve validation Accuracy@1/5, the benchmark falls back to the simpler retrieval model.

## DeepSeek candidate reranking

`DeepSeekCandidateReranker` is separate from the rationale writer.

Candidate reranking:

```text
fixed Top-N codes + terms + query
        -> ranked_codes containing exactly the same codes
```

The response is rejected when:

- a new code appears;
- a supplied code disappears;
- a code is duplicated.

Use `scripts/rerank_frozen_candidates_deepseek.py` after fixing the prompt/model policy on development or validation data. Do not tune the reranking prompt on held-out TEST labels.

External DeepSeek calls remain blocked by the client for restricted/private clinical data. Public/synthetic data still require explicit opt-in.

## Rationale generation is a different stage

After a code is selected, the rationale writer receives a locked code and approved evidence. It cannot silently revise the code. Coding accuracy and rationale quality must be reported separately.

## Explanation quality gate

Every explanation can be classified as `PASS`, `WARN` or `FAIL` using auditable checks:

- at least one evidence quote is present;
- evidence is verbatim in the source text;
- obvious negation/uncertainty/family-history context does not require review;
- terminology support is present;
- retain/remove faithfulness diagnostics are available where supported.

A `FAIL` can only make the operational decision more conservative:

```text
AUTO_CANDIDATE / CODE_PROPOSAL -> HUMAN_REVIEW
```

It can never promote `HUMAN_REVIEW` to automatic handling.

## Recommended comparison table

For a real CADEC run, report at minimum:

```text
Method                         Acc@1   Acc@5   Recall@10   AUTO coverage @ target
Terminology-only lexical
+ historical coding memory
+ biomedical dense retrieval
+ cross-encoder reranking
+ fixed-prompt LLM reranking   (optional, separately reported)
```

Also report seen/unseen codes, candidate-generation versus ranking failures, explanation quality, and expert plausibility review.

No real CADEC or MIMIC performance is implied by the availability of these methods. Results become clinical evidence only after passing the existing dataset readiness and Results Contract gates.
