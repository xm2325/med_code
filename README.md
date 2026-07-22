# MedCode — v0.1.0

MedCode is an explainable clinical-coding research/application toolkit for **MedDRA concept normalisation** and **ICD coding support**.

The main workflow is:

```text
clinical text
    ↓
terminology retrieval + historical expert-coding memory
    ↓
optional biomedical dense retrieval / reranking
    ↓
ranked code candidates
    ↓
uncertainty + OOD + evidence/context checks
    ├── AUTO_CANDIDATE
    ├── TOP_K_HUMAN_CHOICE
    └── FULL_EXPERT_REVIEW
              ↓
for every displayed candidate:
exact source evidence + terminology support + historical provenance + rationale
              ↓
persistent expert review
              ↓
versioned feedback ledger
              ↓
future-release learning memory
              ↓
replayable audit trail
```

## What v0.1.0 means

v0.1.0 is the first **software-complete application release**. It does **not** mean that clinical deployment is validated or that any particular real-data accuracy has already been proven.

The release contract requires:

- audited real-data adapters;
- held-out evaluation and a Results Contract;
- frozen model/policy artifacts;
- uncertainty-aware routing;
- Top-K human choice;
- a separate evidence/rationale object for every displayed candidate;
- persistent expert review;
- a versioned feedback loop that cannot rewrite frozen TEST results;
- replayable audit bundles and decision traces.

See `release/v0.1.0.json` and `docs/V010_RELEASE.md`.

## Core evidence rule

A code is never considered explainable merely because an LLM wrote a plausible paragraph.

For each candidate MedDRA/ICD code, MedCode keeps separate:

1. **Source evidence** — exact verbatim text spans and offsets when available.
2. **Terminology support** — preferred term, synonyms, definition, hierarchy and knowledge source when supplied.
3. **Historical provenance** — similar TRAIN historical expert-coded examples for the same code.
4. **Rationale** — why those supplied facts support that candidate.
5. **Faithfulness/context checks** — whether evidence affects the model score and whether negation/uncertainty/family-history context requires review.

Missing evidence is shown as missing; it must not be invented.

## Uncertain cases: Top-K human choice

MedCode supports three operational routes:

```text
AUTO_CANDIDATE
TOP_K_HUMAN_CHOICE
FULL_EXPERT_REVIEW
```

For `TOP_K_HUMAN_CHOICE`, the reviewer sees up to K alternatives. **Every option** has its own evidence/rationale package, not only the model's first choice.

Build review packets:

```bash
python scripts/build_topk_review_packets.py \
  --predictions outputs/predictions_with_text.csv \
  --terminology /secure/terminology.csv \
  --output-dir outputs/review \
  --top-k 5
```

Optional interactive review UI:

```bash
python -m pip install -e '.[app]'
MEDCODE_REVIEW_DB=review_queue.sqlite3 streamlit run review_app.py
```

The review queue records accepted top-1 codes, alternative Top-K selections, recoding outside Top-K, escalation and no-code decisions.

## Expert feedback without evaluation leakage

Review feedback is stored with:

- model release and frozen-policy identity;
- original Top-K candidate set and original route;
- human final action/code;
- correction reason and hashed reviewer identifier.

Feedback may become **future-release training memory**. It never mutates the historical TEST labels, predictions or metrics of the release that generated the review item.

Useful feedback metrics include top-1 accept rate, Top-K rescue rate and outside-Top-K correction rate.

## Benchmark profiles

### CADEC → MedDRA

Single-label adverse-event/medical concept normalisation.

The audited path preserves original BRAT spans, including discontinuous spans, and checks source-offset integrity before benchmarking.

```bash
python scripts/run_cadec_v0013.py \
  --cadec-root /path/to/CADEC \
  --terminology /secure/authorised_meddra.csv \
  --output-dir outputs/cadec
```

Optional dense/cross-encoder backends must earn selection on VALIDATION before the frozen configuration is evaluated on TEST.

### MedNorm → MedDRA

A public real-data concept-normalisation path is available through `scripts/run_mednorm_real.py`.

Without an authorised full MedDRA terminology file, the runner uses **TRAIN-derived aliases only**. That is an honest closed-code diagnostic: unseen TEST codes are structurally unavailable and the result must not be presented as a full open-set MedDRA benchmark.

With an authorised terminology file:

```bash
python scripts/run_mednorm_real.py \
  --output-dir outputs/mednorm \
  --external-terminology /secure/authorised_meddra.csv
```

The official dataset/licence record remains the source authority; a public mirror may be used only as a transport convenience.

### MIMIC-IV-Note → ICD-10

True multi-label discharge-summary coding with patient-disjoint splitting.

```bash
python scripts/run_mimic_v0012.py \
  --records /secure/derived/mimic_icd10_records.csv \
  --terminology /secure/derived/icd10_terminology.csv \
  --output-dir /secure/results/mimic
```

A target proposal precision applies to **individual code proposals**, not to the percentage of complete notes that can be automatically coded.

## DeepSeek integration

DeepSeek has two deliberately separate roles:

**Candidate reranking**

```text
frozen Top-N candidate codes
        ↓
DeepSeek
        ↓
same exact code set, reordered only
```

Responses that add, remove or duplicate candidate codes are rejected.

**Rationale writing**

```text
locked selected/candidate code
+
approved verbatim evidence
+
terminology knowledge
        ↓
DeepSeek rationale
```

The code is locked and returned evidence must match the approved source evidence. The provided external client blocks `restricted`/`private` data by default; do not send governed BSRBR/MIMIC clinical text to an external endpoint unless governance explicitly permits it.

`DEEPSEEK_API_KEY` is read from environment/GitHub secrets only and is never committed.

## Evaluation outputs

Depending on the benchmark, MedCode reports:

- Accuracy@1/@K or multi-label F1/precision/recall;
- candidate recall@K;
- seen vs unseen-code performance;
- candidate-generation vs ranking failures;
- terminology-only vs historical-memory value;
- validation-selected coverage/workload trade-offs;
- uncertainty route counts;
- explanation grounded/verbatim/quality rates;
- expert feedback rescue/correction rates;
- Results Contract reportability status.

Do not pool CADEC Accuracy@1 and MIMIC multi-label F1 into one score; they are different prediction tasks.

## Audit and replay

A release can build an `audit_bundle.json` containing SHA256 fingerprints for metrics, predictions, frozen policy, Results Contract and experiment manifest.

```bash
python scripts/build_audit_bundle.py --benchmark-dir outputs/benchmark --release 0.1.0
```

A decision trace can record:

```text
model prediction
→ uncertainty
→ explanation quality
→ route before review
→ human review event
→ final code/status
```

This makes later review possible without silently rewriting historical outputs.

## Release vs clinical readiness

`software_release_complete = true` means the v0.1 software capabilities are present.

Clinical or operational deployment additionally requires, at minimum, appropriate data governance, authorised terminology resources, real held-out human-reference evaluation, prospective validation, expert review of rationale plausibility, calibration/monitoring and local safety approval.

No real CADEC, MedNorm, MIMIC, BSRBR, ICD-10 or MedDRA performance number is implied by the code alone.
