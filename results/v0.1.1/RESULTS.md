# MedCode v0.1.1 — real MedNorm evaluation

> This is a real-data **closed-code** evaluation using public MedNorm-derived phrases and TRAIN-derived candidate aliases. It is not a full licensed-MedDRA open-set benchmark.

## Observed results

| Metric | Result |
|---|---:|
| Real baseline test sample | 500 cases |
| Baseline Accuracy@1 | 57.0% |
| Baseline Accuracy@3 | 68.0% |
| Baseline / candidate Recall@5 | 71.8% |
| Gold code seen in TRAIN rate | 95.4% |
| Paired fixed-seed subset | 24 cases |
| Paired baseline Accuracy@1 | 41.7% |
| DeepSeek API calls attempted | 24 |
| Valid DeepSeek responses | 19 |
| DeepSeek accepted-only Accuracy@1 | 63.2% |
| DeepSeek accepted-only Accuracy@1 delta | 15.8% |
| Fixed candidate Recall@5 | 70.8% |
| Valid DeepSeek API response rate | 79.2% |

`Recall@5` is also an **oracle Top-K human-choice upper bound**: it assumes a human always chooses the gold code whenever present and is not observed human-assisted accuracy.

## Routing

```json
{
  "TOP_K_HUMAN_CHOICE": 305,
  "AUTO_CANDIDATE": 195
}
```

## Candidate rationale contract

Every displayed option is tied to the real benchmark phrase. DeepSeek output, when available, must preserve the exact fixed candidate code set and provide one rationale per candidate citing exact approved real-data evidence. When DeepSeek is unavailable, every option still carries deterministic grounded evidence/rationale and is explicitly labelled as such.

## Example real-data cases

### ARTHROTEC.13.ann_TT7

**Real phrase:** `menstrual cramps without vaginal bleeding`  
**Gold:** `10013935` · **Baseline top1:** `10055798` · **DeepSeek top1:** `not run / no valid response` · **Route:** `TOP_K_HUMAN_CHOICE`

- **#1 `10055798` — bleeding**
  - Evidence: `menstrual cramps without vaginal bleeding`
  - Rationale (deterministic_grounded): Source evidence: “menstrual cramps without vaginal bleeding” Terminology mapping: 10055798 — bleeding. 1 similar TRAIN historical expert-coded example(s) support this code as provenance.
- **#2 `10028334` — Night cramps**
  - Evidence: `menstrual cramps without vaginal bleeding`
  - Rationale (deterministic_grounded): Source evidence: “menstrual cramps without vaginal bleeding” Terminology mapping: 10028334 — Night cramps. 2 similar TRAIN historical expert-coded example(s) support this code as provenance.
- **#3 `10022437` — Sleeplessness**
  - Evidence: `menstrual cramps without vaginal bleeding`
  - Rationale (deterministic_grounded): Source evidence: “menstrual cramps without vaginal bleeding” Terminology mapping: 10022437 — Sleeplessness. 1 similar TRAIN historical expert-coded example(s) support this code as provenance.
- **#4 `10000081` — Abdominal Cramps**
  - Evidence: `menstrual cramps without vaginal bleeding`
  - Rationale (deterministic_grounded): Source evidence: “menstrual cramps without vaginal bleeding” Terminology mapping: 10000081 — Abdominal Cramps. 1 similar TRAIN historical expert-coded example(s) support this code as provenance.
- **#5 `10046910` — Vaginal Hemorrhage**
  - Evidence: `menstrual cramps without vaginal bleeding`
  - Rationale (deterministic_grounded): Source evidence: “menstrual cramps without vaginal bleeding” Terminology mapping: 10046910 — Vaginal Hemorrhage.

### LIPITOR.831.ann_TT2

**Real phrase:** `pain in my arm and shoulder`  
**Gold:** `10033425` · **Baseline top1:** `10028391` · **DeepSeek top1:** `10028391` · **Route:** `AUTO_CANDIDATE`

- **#1 `10028391` — Musculoskeletal Pain**
  - Evidence: `pain in my arm and shoulder`
  - Rationale (deepseek_validated): Directly matches via synonyms that explicitly combine shoulder and arm pain, such as 'shoulders and arms pain' and 'Shoulder Pain', making it the most specific and relevant code.
- **#3 `10033371` — Acute onset pain**
  - Evidence: `pain in my arm and shoulder`
  - Rationale (deepseek_validated): Captures the pain symptom generally, but the phrase does not mention onset or chronicity, and the code's specificity to 'acute onset' is not supported. The match is only at the broad 'pain' level, making it less targeted.
- **#5 `10048010` — Withdrawal Symptoms**
  - Evidence: `pain in my arm and shoulder`
  - Rationale (deepseek_validated): Completely unrelated; withdrawal symptoms are not mentioned or implied in the phrase 'pain in my arm and shoulder', and no evidence supports this context.
- **#2 `10033425` — Pain in limb**
  - Evidence: `pain in my arm and shoulder`
  - Rationale (deepseek_validated): Relevant because 'Pain in limb' includes synonyms like 'Arm Pain' and 'pains down arms', which partially match the arm component. However, 'shoulder' is not explicitly listed, and the phrase mentions both arm and shoulder, so it is less precise than codes that include shoulder.
- **#4 `10003239` — Coxalgia**
  - Evidence: `pain in my arm and shoulder`
  - Rationale (deepseek_validated): Coxalgia primarily refers to hip pain, not arm or shoulder. Although synonyms include 'Arthralgia' (joint pain), the code is anatomically mismatched, and the phrase does not indicate joint-specific pain.

### LIPITOR.103.ann_TT1

**Real phrase:** `Muscle aches`  
**Gold:** `10028411` · **Baseline top1:** `10033371` · **DeepSeek top1:** `not run / no valid response` · **Route:** `TOP_K_HUMAN_CHOICE`

- **#1 `10033371` — Acute onset pain**
  - Evidence: `Muscle aches`
  - Rationale (deterministic_grounded): Source evidence: “Muscle aches” Terminology mapping: 10033371 — Acute onset pain. 2 similar TRAIN historical expert-coded example(s) support this code as provenance.
- **#2 `10049565` — Muscle Fatigue**
  - Evidence: `Muscle aches`
  - Rationale (deterministic_grounded): Source evidence: “Muscle aches” Terminology mapping: 10049565 — Muscle Fatigue. 1 similar TRAIN historical expert-coded example(s) support this code as provenance.
- **#3 `10028334` — Night cramps**
  - Evidence: `Muscle aches`
  - Rationale (deterministic_grounded): Source evidence: “Muscle aches” Terminology mapping: 10028334 — Night cramps. 2 similar TRAIN historical expert-coded example(s) support this code as provenance.
- **#4 `10028289` — Muscular Atrophy**
  - Evidence: `Muscle aches`
  - Rationale (deterministic_grounded): Source evidence: “Muscle aches” Terminology mapping: 10028289 — Muscular Atrophy.
- **#5 `10028309` — Muscle hemorrhage**
  - Evidence: `Muscle aches`
  - Rationale (deterministic_grounded): Source evidence: “Muscle aches” Terminology mapping: 10028309 — Muscle hemorrhage.

### ARTHROTEC.46.ann_TT1

**Real phrase:** `Naseua`  
**Gold:** `10028813` · **Baseline top1:** `10028735` · **DeepSeek top1:** `10028813` · **Route:** `TOP_K_HUMAN_CHOICE`

- **#2 `10028735` — Nasal congestion (finding)**
  - Evidence: `Naseua`
  - Rationale (deepseek_validated): The phrase "Naseua" does not denote nasal congestion; it contains the substring "nas" but is a misspelling of nausea, not related to stuffy nose. The evidence "Naseua" has no clinical association with nasal congestion.
- **#3 `10028810` — Common Cold**
  - Evidence: `Naseua`
  - Rationale (deepseek_validated): "Naseua" does not correspond to common cold or nasopharyngitis; the phrase is a misspelling of nausea, not a cold. The evidence "Naseua" lacks any link to cold symptoms.
- **#4 `10028765` — Perforation of nasal septum**
  - Evidence: `Naseua`
  - Rationale (deepseek_validated): "Naseua" does not refer to nasal septum perforation; the phrase is unrelated to nasal injuries. The evidence "Naseua" provides no support.
- **#1 `10028813` — Nausea**
  - Evidence: `Naseua`
  - Rationale (deepseek_validated): The phrase "Naseua" is a common misspelling of "nausea," as evidenced by the synonym "nuasea" in the terminology. The direct match to the term "Nausea" makes this the most appropriate code.
- **#5 `10078904` — Pseudotumor Cerebri**
  - Evidence: `Naseua`
  - Rationale (deepseek_validated): "Naseua" does not indicate pseudotumor cerebri; there is no similarity in phrasing or clinical presentation. The evidence "Naseua" is insufficient to justify this code.

### ARTHROTEC.99.ann_TT3

**Real phrase:** `pain`  
**Gold:** `10033371` · **Baseline top1:** `10033371` · **DeepSeek top1:** `10033371` · **Route:** `AUTO_CANDIDATE`

- **#1 `10033371` — Acute onset pain**
  - Evidence: `pain`
  - Rationale (deepseek_validated): Direct match: term 'Acute onset pain' contains 'pain' and synonyms include 'pain', aligning with the sole evidence token.
- **#2 `10019211` — Headache**
  - Evidence: `pain`
  - Rationale (deepseek_validated): Mismatch: 'Headache' specifies a head location, whereas the evidence only supports general 'pain' without anatomical qualifier.
- **#3 `10015958` — Sore eye**
  - Evidence: `pain`
  - Rationale (deepseek_validated): Mismatch: 'Sore eye' is specific to ocular pain; the evidence 'pain' provides no anatomical detail.
- **#4 `10033425` — Pain in limb**
  - Evidence: `pain`
  - Rationale (deepseek_validated): Mismatch: 'Pain in limb' indicates extremity pain; the evidence is simply 'pain' with no limb mention.
- **#5 `10003988` — Back Pain**
  - Evidence: `pain`
  - Rationale (deepseek_validated): Mismatch: 'Back Pain' localizes to the back, while the evidence 'pain' lacks anatomical specificity.

### LIPITOR.501.ann_TT6

**Real phrase:** `Depression`  
**Gold:** `10012378` · **Baseline top1:** `10012378` · **DeepSeek top1:** `not run / no valid response` · **Route:** `AUTO_CANDIDATE`

- **#1 `10012378` — Sad mood**
  - Evidence: `Depression`
  - Rationale (deterministic_grounded): Source evidence: “Depression” Terminology mapping: 10012378 — Sad mood. 2 similar TRAIN historical expert-coded example(s) support this code as provenance.
- **#2 `10057840` — Major Depressive Disorder**
  - Evidence: `Depression`
  - Rationale (deterministic_grounded): Source evidence: “Depression” Terminology mapping: 10057840 — Major Depressive Disorder.
- **#3 `10004939` — Bipolar I disorder**
  - Evidence: `Depression`
  - Rationale (deterministic_grounded): Source evidence: “Depression” Terminology mapping: 10004939 — Bipolar I disorder.
- **#4 `10054089` — Depressive Symptoms**
  - Evidence: `Depression`
  - Rationale (deterministic_grounded): Source evidence: “Depression” Terminology mapping: 10054089 — Depressive Symptoms.
- **#5 `10012374` — Depressed mood**
  - Evidence: `Depression`
  - Rationale (deterministic_grounded): Source evidence: “Depression” Terminology mapping: 10012374 — Depressed mood.

### VOLTAREN-XR.3.ann_TT3

**Real phrase:** `chest discomfort`  
**Gold:** `10008469` · **Baseline top1:** `10053155` · **DeepSeek top1:** `10008469` · **Route:** `TOP_K_HUMAN_CHOICE`

- **#3 `10053155` — Gastric irritation**
  - Evidence: `chest discomfort`
  - Rationale (deepseek_validated): Although 'chest discomfort' can be reported with gastric irritation, the code 'Gastric irritation' primarily denotes stomach upset, not a chest symptom.
- **#2 `10008479` — Retrosternal pain**
  - Evidence: `chest discomfort`
  - Rationale (deepseek_validated): 'Retrosternal pain' is a specific type of chest discomfort, directly linked to the symptom 'chest discomfort'.
- **#4 `10024968` — chest infections**
  - Evidence: `chest discomfort`
  - Rationale (deepseek_validated): While 'chest infections' can cause 'chest discomfort', the code represents an infectious condition rather than the symptom itself.
- **#1 `10008469` — Chest discomfort**
  - Evidence: `chest discomfort`
  - Rationale (deepseek_validated): The term 'Chest discomfort' is an exact lexical match for the benchmark phrase, with synonyms including 'chest discomfort'.
- **#5 `10019030` — Hair discoloration**
  - Evidence: `chest discomfort`
  - Rationale (deepseek_validated): 'Hair discoloration' is unrelated to 'chest discomfort' and is an implausible mapping.

### LIPITOR.773.ann_TT2

**Real phrase:** `extremly bad pains in arms`  
**Gold:** `10033371` · **Baseline top1:** `10033425` · **DeepSeek top1:** `10033425` · **Route:** `TOP_K_HUMAN_CHOICE`

- **#1 `10033425` — Pain in limb**
  - Evidence: `extremly bad pains in arms`
  - Rationale (deepseek_validated): Directly matches: 'pains in arms' indicates pain in a limb; 'Pain in limb' precisely captures the anatomical location.
- **#2 `10028391` — Musculoskeletal Pain**
  - Evidence: `extremly bad pains in arms`
  - Rationale (deepseek_validated): Partially matches: arm pain may be musculoskeletal, but term includes shoulder pain not mentioned; less specific than 'Pain in limb'.
- **#3 `10033371` — Acute onset pain**
  - Evidence: `extremly bad pains in arms`
  - Rationale (deepseek_validated): Mismatch: phrase specifies arms, but 'Acute onset pain' lacks anatomical reference; severity alone insufficient without site.
- **#4 `10000087` — Abdominal pain upper**
  - Evidence: `extremly bad pains in arms`
  - Rationale (deepseek_validated): No match: 'Abdominal pain upper' refers to stomach pain, unrelated to 'pains in arms'.
- **#5 `10022998` — Irritable Mood**
  - Evidence: `extremly bad pains in arms`
  - Rationale (deepseek_validated): No match: 'Irritable Mood' is a mood disorder, not a pain condition; phrase clearly describes physical pain.

## Data/source note

MedNorm official source/licence authority: DOI `10.17632/b9x7xxb9sz.1`, CC BY-NC 3.0. The Hugging Face copy is used only as a transport convenience and does not override the official licence.