# RA hidden-comorbidity recovery track

## Question

For patients with known rheumatoid arthritis (RA), which clinically supported comorbidities are incompletely represented in structured coding, and which can be recovered from clinical free text with grounded evidence and acceptable false-positive risk?

This track is deliberately different from generic `note -> ICD` coding.

## Three-source evaluation

For each patient/admission × phenotype:

- `G`: clinician/reference phenotype.
- `C`: structured-code signal.
- `T`: text-derived model call.

The main recovery cell is `G=1, C=0, T=1`.

The primary metric is the Hidden Comorbidity Recovery Rate (HCRR):

```text
HCRR = N(G=1,C=0,T=1) / N(G=1,C=0)
```

Also report the PPV of `T=1,C=0` candidates, code/text recall against the reference, combined `C OR T` recall, and an error taxonomy covering negation, family history, uncertainty, historical/resolved disease, wrong concept and wrong temporal state.

## MIPA role

MIPA is used as a controlled method benchmark, not as an unbiased estimate of RA comorbidity prevalence. Its candidate discharge summaries were enriched using phenotype-related ICD signals before multilabel expert annotation.

The public labels reproduce:

- 1,388 admissions in the benchmark;
- 164 RA-positive admissions;
- 99 unique RA patients after de-duplication.

Patient-level multimorbidity summaries must group repeated admissions by `subject_id`.

## First target panel

1. **Past DVT/PE (VTE history): primary discovery target.** Historical status is narrative/temporal, and the public MIPA benchmark shows a large gap between ICD-only and LLM text phenotyping.
2. **Depression: necessity test.** A simple TF-IDF baseline is strong, so an LLM must show added value rather than being assumed necessary.
3. **Hypertension: high-volume target.** Useful for sample size, but structured information is relatively informative.
4. **Obesity: secondary discovery target.** Evidence can be expressed through BMI or narrative.
5. **Type 2 diabetes and HFpEF: structured-signal controls.** They test whether text is adding information where structured coding already works well.

## Acceptance gates

### Gate A — public feasibility

Pass when public labels reproduce dataset/RA counts, repeated admissions are separated from unique patients, and there are enough positive examples for at least one difficult target and one structured control.

### Gate B — real Gold × Code × Text benchmark

Pass when real structured-code calls and real text calls exist for the same cases, all splits are patient-disjoint, HCRR is calculable, and every text-positive proposal includes source evidence.

### Gate C — open-ended discovery

Pass when the system can propose conditions outside a fixed phenotype list, explicitly models assertion/temporality, and 50–100 text-positive/code-negative candidates (or all if fewer) are manually adjudicated with PPV and error categories reported.

### Gate D — BSRBR-RA transfer

Pass when historical free text, structured comorbidity data and expert MedDRA coding are joined at the correct patient/event/time level; evaluation uses temporal patient-level splits; exact/semantic code agreement, top-K coverage, calibration, abstention and review workload are reported; and uncertain cases are routed to human review.

## Stop/redirect rule

Do not force a positive result. Drop or reframe a phenotype when the gold-positive/code-negative set is too small, text-only candidate PPV is poor, or a simple baseline matches the LLM. A valid result is that text adds value only for context- or time-sensitive comorbidities while structured coding is sufficient for others.
