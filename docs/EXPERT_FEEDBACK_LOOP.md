# Expert feedback loop — v0.0.15

Human review is recorded as a versioned ledger rather than silently overwriting model predictions.

Each event records the model release, frozen policy, original Top-K candidate set, original route, human action, final selected code and a review reason. Reviewer identifiers can be hashed.

Supported actions:

- `ACCEPT_TOP1`
- `SELECT_ALTERNATIVE`
- `RECODE_OUTSIDE_TOPK`
- `ESCALATE`
- `NO_CODE`

Feedback can be transformed into a **new future-release training-memory artifact**. It must never mutate the already-frozen TEST labels, predictions, metrics or results contract of the release that generated the review item.

Useful operational metrics include top-1 accept rate, Top-K rescue rate, and outside-Top-K correction rate. A high outside-Top-K rate indicates a candidate-generation problem rather than a human-selection problem.
