# Reranking Protocol

Candidate generation and candidate reranking are evaluated separately.

## Frozen candidate protocol

1. Generate the top-N terminology candidates without access to TEST labels.
2. Save candidate codes, terms, retrieval scores, clinical mention/context, and a stable prediction ID.
3. Apply a reranker only to this frozen candidate set.
4. The reranker must not invent codes outside the supplied candidate set for this experiment.
5. Evaluate all rerankers against the same held-out reference labels.

## Comparable methods

- lexical terminology retrieval;
- dense biomedical retrieval when a permitted model is available;
- historical coding-memory retrieval;
- hybrid terminology + historical retrieval;
- local cross-encoder reranking;
- external LLM reranking through JSONL export/import.

## Diagnostics

Report candidate recall at k separately from final Accuracy@1. If the correct code is absent from the candidate set, classify the error as candidate-generation failure. If it is present but not ranked first, classify it as ranking/disambiguation failure.

## LLM boundary

For a controlled reranking experiment, provide only the frozen candidate codes/terms plus the allowed clinical context. Require structured output containing the prediction ID and ranked candidate codes. Reject or flag outputs containing codes outside the candidate set.

Do not tune prompts using TEST labels.
