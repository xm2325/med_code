# Uncertainty-aware Top-K review — v0.0.14

MedCode does not force every case into a single automatic code.

```text
clinical text
    ↓
ranked code candidates
    ↓
uncertainty + OOD + explanation audit
    ├── AUTO_CANDIDATE
    ├── TOP_K_HUMAN_CHOICE
    └── FULL_EXPERT_REVIEW
```

## Top-K human choice

For uncertain-but-informative cases, the reviewer receives up to five candidate codes. **Every displayed option has its own evidence and rationale**, not only the model's first choice.

Each option contains:

- code and preferred term;
- model score/rank;
- verbatim source evidence spans where they can be grounded;
- terminology term/synonyms/definition/hierarchy and knowledge source when supplied;
- similar TRAIN historical expert-coded cases for that same code when available;
- an explicit deterministic rationale explaining what supports the candidate;
- an explicit statement when no exact evidence span could be grounded.

Missing evidence is never replaced by an invented quote.

## Routing semantics

`AUTO_CANDIDATE` means the frozen coding policy is sufficiently confident **and** uncertainty/context/evidence checks do not force review.

`TOP_K_HUMAN_CHOICE` means the system has useful ranked candidates but should not decide alone. The human can select one candidate or escalate.

`FULL_EXPERT_REVIEW` means the model evidence is too weak/OOD, or a safety/explanation gate failed.

The uncertainty features are development diagnostics, not calibrated clinical probabilities. The routing policy must be selected and validated on development/validation data before prospective use.
