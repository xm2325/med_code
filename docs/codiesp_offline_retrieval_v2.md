# CodiEsp offline ICD retrieval v2

## Why this stage exists

The first 10-case Direct-vs-RAG pilot showed that the current whole-note lexical ICD retrieval can inject irrelevant ICD neighbourhoods into the LLM context. Before spending more DeepSeek/API budget, retrieval itself must be validated independently.

This stage therefore makes **zero LLM/API calls**. It uses CodiEsp **DEV** only for method development and reports retrieval gold-code Recall@5/10/20/35.

## Implemented methods

1. `whole_note_lexical`
   - Whole clinical note as one query.
   - Sparse BM25 + character TF-IDF.
   - This is the closest offline control for the current lexical approach.

2. `sentence_hybrid`
   - Whole-note signal plus sentence-level max pooling.
   - BM25 + character TF-IDF + offline latent semantic analysis (TF-IDF + TruncatedSVD).

3. `fragment_hybrid_alias`
   - Deterministic evidence-fragment decomposition: sentences, selected clauses, and cue-triggered spans.
   - Query-side abbreviation expansion.
   - KB-side official/CodiEsp descriptions, ICD notes/inclusion terms, and hierarchy text.
   - Hybrid BM25 + char TF-IDF + latent semantic score.

4. `fragment_hybrid_alias_hierarchy`
   - Same as method 3.
   - Adds post-retrieval ICD hierarchy neighbourhood expansion with score decay.

The phrase **fragment decomposition** is deliberate: this implementation is deterministic and should not be described as a validated clinical NER model.

## Scientific split policy

- Default: `DEV`.
- `TEST` is blocked unless `--allow-test-eval` is explicitly supplied.
- Retrieval method/weights/thresholds must be frozen on DEV before TEST is touched.
- Do not select a retrieval method using TEST gold labels.

## Primary acceptance metrics

The retrieval stage is evaluated before any LLM:

- micro Recall@5
- micro Recall@10
- micro Recall@20
- micro Recall@35
- macro per-case Recall@K
- case hit-rate@K
- miss categories

Selection rule in the initial implementation:

> maximise DEV micro Recall@20; tie-break by DEV micro Recall@35.

Recall is prioritised because a gold code absent from the retrieved knowledge cannot benefit from downstream RAG validation. Precision/compression will be addressed by reranking and code-specific validation after high-recall retrieval is established.

## Miss diagnostics

Missed gold codes are classified into transparent heuristic categories:

- `hierarchy_near_miss`
- `description_present_but_not_ranked`
- `lexical_signal_ranked_low`
- `weak_lexical_signal`
- `synonym_translation_or_inference_gap`

These are diagnostic categories, not gold clinical error labels.

## Run locally

```bash
python scripts/run_codiesp_offline_retrieval_benchmark.py \
  --split dev \
  --output-dir results/codiesp_offline_retrieval/dev
```

Optional deterministic smoke run:

```bash
python scripts/run_codiesp_offline_retrieval_benchmark.py \
  --split dev \
  --limit 25 \
  --output-dir results/codiesp_offline_retrieval/dev25
```

The script downloads only the public CodiEsp/CDC resources already used by the repository. It does **not** read `DEEPSEEK_API_KEY` and does not call DeepSeek or any LLM API.

## Outputs

- `retrieval_summary.csv`
- `per_case_retrieval.csv`
- `retrieval_misses.csv`
- `retrieval_miss_categories.csv`
- `retrieval_manifest.json`
- `selected_retrieval_config.json`

## Gate before any new DeepSeek experiment

Do **not** launch another paid Direct-vs-RAG LLM experiment merely because this script exists.

Proceed to the next stage only if the DEV retrieval benchmark shows a meaningful improvement over `whole_note_lexical`, especially at Recall@20/35, and the miss analysis indicates fewer lexical-collision/synonym-gap failures.

Then freeze the retrieval configuration and build the next pipeline:

`clinical text -> evidence/condition decomposition -> entity-level retrieval -> reranking -> ICD hierarchy/rule validation -> LLM audit`

The LLM should be used only after the deterministic retrieval stage passes this gate.
