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
| Valid DeepSeek responses | 24 |
| DeepSeek accepted-only Accuracy@1 | 54.2% |
| DeepSeek accepted-only Accuracy@1 delta | 12.5% |
| Fixed candidate Recall@5 | 70.8% |
| Valid DeepSeek API response rate | 100.0% |

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
**Gold:** `10013935` · **Baseline top1:** `10055798` · **DeepSeek top1:** `10000081` · **Route:** `TOP_K_HUMAN_CHOICE`

- **#3 `10055798` — bleeding**
  - Evidence: `menstrual cramps without vaginal bleeding`
  - Rationale (deepseek_validated): The phrase explicitly states 'without vaginal bleeding', negating the presence of bleeding. The code 'bleeding' directly contradicts this negation, making it a mismatch. The evidence 'menstrual cramps without vaginal bleeding' clearly indicates absence of bleeding.
- **#2 `10028334` — Night cramps**
  - Evidence: `menstrual cramps without vaginal bleeding`
  - Rationale (deepseek_validated): The phrase mentions 'cramps', which aligns with 'Night cramps' as a type of cramp. However, 'Night cramps' specifies nocturnal occurrence and is not specific to menstruation, making it a weaker match. The evidence 'menstrual cramps without vaginal bleeding' includes the word 'cramps'.
- **#5 `10022437` — Sleeplessness**
  - Evidence: `menstrual cramps without vaginal bleeding`
  - Rationale (deepseek_validated): The phrase 'menstrual cramps without vaginal bleeding' does not mention sleep or sleeplessness. 'Sleeplessness' is unrelated to the described symptoms. The evidence provides no support for this code.
- **#1 `10000081` — Abdominal Cramps**
  - Evidence: `menstrual cramps without vaginal bleeding`
  - Rationale (deepseek_validated): The phrase 'menstrual cramps without vaginal bleeding' indicates abdominal cramping associated with menstruation. 'Abdominal Cramps' directly matches the symptom of cramps, though it does not specify menstrual origin. The evidence 'menstrual cramps without vaginal bleeding' supports that cramps are present.
- **#4 `10046910` — Vaginal Hemorrhage**
  - Evidence: `menstrual cramps without vaginal bleeding`
  - Rationale (deepseek_validated): The phrase specifies 'without vaginal bleeding', while 'Vaginal Hemorrhage' denotes heavy vaginal bleeding, which is the opposite of the stated condition. This is a clear mismatch. The evidence 'menstrual cramps without vaginal bleeding' explicitly excludes vaginal bleeding.

### LIPITOR.831.ann_TT2

**Real phrase:** `pain in my arm and shoulder`  
**Gold:** `10033425` · **Baseline top1:** `10028391` · **DeepSeek top1:** `10028391` · **Route:** `AUTO_CANDIDATE`

- **#1 `10028391` — Musculoskeletal Pain**
  - Evidence: `pain in my arm and shoulder`
  - Rationale (deepseek_validated): Strong match: The phrase 'pain in my arm and shoulder' aligns with the synonym 'shoulders and arms pain' in the terminology support, indicating musculoskeletal pain in the shoulder and arm region.
- **#3 `10033371` — Acute onset pain**
  - Evidence: `pain in my arm and shoulder`
  - Rationale (deepseek_validated): Weak match: The phrase describes pain, but 'Acute onset pain' is a general pain term without specifying location or acuteness. The evidence does not indicate acute onset, so this code is too broad.
- **#5 `10048010` — Withdrawal Symptoms**
  - Evidence: `pain in my arm and shoulder`
  - Rationale (deepseek_validated): Mismatch: 'Withdrawal Symptoms' relates to substance withdrawal, with synonyms like 'withdrawal symptoms' and 'drug withdrawal syndrome'. The phrase does not mention any withdrawal context, making this code irrelevant.
- **#2 `10033425` — Pain in limb**
  - Evidence: `pain in my arm and shoulder`
  - Rationale (deepseek_validated): Partial match: The phrase mentions arm pain, which is a limb, and the terminology includes 'Arm Pain' and 'pains down arms'. However, it does not explicitly cover shoulder pain, making it less specific than 'Musculoskeletal Pain'.
- **#4 `10003239` — Coxalgia**
  - Evidence: `pain in my arm and shoulder`
  - Rationale (deepseek_validated): Mismatch: 'Coxalgia' refers to hip pain, as seen in synonyms like 'hip pain' and 'hip ache'. The phrase specifies arm and shoulder pain, not hip, so this code is incorrect.

### LIPITOR.103.ann_TT1

**Real phrase:** `Muscle aches`  
**Gold:** `10028411` · **Baseline top1:** `10033371` · **DeepSeek top1:** `10033371` · **Route:** `TOP_K_HUMAN_CHOICE`

- **#1 `10033371` — Acute onset pain**
  - Evidence: `Muscle aches`
  - Rationale (deepseek_validated): The term 'Acute onset pain' includes synonyms like 'aches' and 'pain', which directly match the 'aches' in 'Muscle aches'. The evidence 'Muscle aches' supports this as a general pain code.
- **#2 `10049565` — Muscle Fatigue**
  - Evidence: `Muscle aches`
  - Rationale (deepseek_validated): 'Muscle Fatigue' is related to muscle discomfort but does not explicitly denote pain or aching. The evidence 'Muscle aches' suggests pain rather than fatigue, making this a weaker match.
- **#3 `10028334` — Night cramps**
  - Evidence: `Muscle aches`
  - Rationale (deepseek_validated): 'Night cramps' involves muscle spasms, which can be painful but are distinct from a general ache. The evidence 'Muscle aches' does not specify cramping, so this is a partial match at best.
- **#4 `10028289` — Muscular Atrophy**
  - Evidence: `Muscle aches`
  - Rationale (deepseek_validated): 'Muscular Atrophy' refers to muscle wasting, not pain. The evidence 'Muscle aches' indicates pain, not atrophy, so this code is a poor match.
- **#5 `10028309` — Muscle hemorrhage**
  - Evidence: `Muscle aches`
  - Rationale (deepseek_validated): 'Muscle hemorrhage' is bleeding into muscle, which is a specific injury, not a general ache. The evidence 'Muscle aches' does not suggest hemorrhage, making this an unlikely match.

### ARTHROTEC.46.ann_TT1

**Real phrase:** `Naseua`  
**Gold:** `10028813` · **Baseline top1:** `10028735` · **DeepSeek top1:** `10028813` · **Route:** `TOP_K_HUMAN_CHOICE`

- **#2 `10028735` — Nasal congestion (finding)**
  - Evidence: `Naseua`
  - Rationale (deepseek_validated): The phrase 'Naseua' could be a misspelling of 'Nasal', but the code 'Nasal congestion (finding)' does not directly match the intended meaning. The allowed_source_evidence 'Naseua' is not a known synonym for nasal congestion, and the retrieval score is low, indicating a weak match.
- **#3 `10028810` — Common Cold**
  - Evidence: `Naseua`
  - Rationale (deepseek_validated): The phrase 'Naseua' might be a misspelling of 'Nasopharyngitis', but the code 'Common Cold' is a broader term. The allowed_source_evidence 'Naseua' does not closely match any synonym for this code, and the retrieval score is low, suggesting a poor match.
- **#4 `10028765` — Perforation of nasal septum**
  - Evidence: `Naseua`
  - Rationale (deepseek_validated): The phrase 'Naseua' does not relate to 'Perforation of nasal septum'. The allowed_source_evidence 'Naseua' has no connection to this code, and the retrieval score is low, indicating a mismatch.
- **#1 `10028813` — Nausea**
  - Evidence: `Naseua`
  - Rationale (deepseek_validated): The benchmark phrase 'Naseua' is a common misspelling of 'Nausea', which is explicitly listed as a synonym for this code: 'nuasea' is a close variant. The allowed_source_evidence contains 'Naseua', which matches the misspelling pattern.
- **#5 `10078904` — Pseudotumor Cerebri**
  - Evidence: `Naseua`
  - Rationale (deepseek_validated): The phrase 'Naseua' has no relation to 'Pseudotumor Cerebri'. The allowed_source_evidence 'Naseua' does not match any synonym for this code, and the retrieval score is very low, indicating a clear mismatch.

### ARTHROTEC.99.ann_TT3

**Real phrase:** `pain`  
**Gold:** `10033371` · **Baseline top1:** `10033371` · **DeepSeek top1:** `10033371` · **Route:** `AUTO_CANDIDATE`

- **#1 `10033371` — Acute onset pain**
  - Evidence: `pain`
  - Rationale (deepseek_validated): The term 'Acute onset pain' directly contains the word 'pain', and the allowed_source_evidence includes 'pain' as an exact match. This is the most general and direct mapping for the benchmark phrase.
- **#2 `10019211` — Headache**
  - Evidence: `pain`
  - Rationale (deepseek_validated): The term 'Headache' is a specific type of pain, but the benchmark phrase 'pain' does not specify location. The allowed_source_evidence only contains 'pain', which is a broader concept. This code is a plausible but less specific match.
- **#3 `10015958` — Sore eye**
  - Evidence: `pain`
  - Rationale (deepseek_validated): The term 'Sore eye' refers to pain in the eye, but the benchmark phrase 'pain' does not indicate location. The allowed_source_evidence 'pain' is a general term, making this code a possible but overly specific match.
- **#4 `10033425` — Pain in limb**
  - Evidence: `pain`
  - Rationale (deepseek_validated): The term 'Pain in limb' specifies pain in a limb, but the benchmark phrase 'pain' lacks location details. The allowed_source_evidence 'pain' is a broader concept, so this code is a plausible but less direct match.
- **#5 `10003988` — Back Pain**
  - Evidence: `pain`
  - Rationale (deepseek_validated): The term 'Back Pain' indicates pain in the back, but the benchmark phrase 'pain' does not specify location. The allowed_source_evidence 'pain' is a general term, making this code a possible but overly specific match.

### LIPITOR.501.ann_TT6

**Real phrase:** `Depression`  
**Gold:** `10012378` · **Baseline top1:** `10012378` · **DeepSeek top1:** `10012378` · **Route:** `AUTO_CANDIDATE`

- **#1 `10012378` — Sad mood**
  - Evidence: `Depression`
  - Rationale (deepseek_validated): The term 'Sad mood' is directly supported by the allowed_source_evidence containing the exact string 'Depression', which is a synonym for this code. This is the closest lexical match to the benchmark phrase.
- **#2 `10057840` — Major Depressive Disorder**
  - Evidence: `Depression`
  - Rationale (deepseek_validated): Major Depressive Disorder is a specific clinical diagnosis that encompasses depression, but the benchmark phrase 'Depression' is broader and does not explicitly indicate a disorder. The evidence includes 'Depression' as a synonym, but the term 'Major Depressive Disorder' is more specific.
- **#5 `10004939` — Bipolar I disorder**
  - Evidence: `Depression`
  - Rationale (deepseek_validated): Bipolar I disorder includes depressive episodes, but it is a distinct condition from unipolar depression. The evidence includes 'Depression, Bipolar' as a synonym, but the benchmark phrase 'Depression' alone does not imply bipolarity. This is a weaker match.
- **#3 `10054089` — Depressive Symptoms**
  - Evidence: `Depression`
  - Rationale (deepseek_validated): Depressive Symptoms refers to the manifestations of depression, which is related but not identical to the benchmark phrase 'Depression'. The evidence includes 'depression symptoms' as a synonym, indicating a partial match.
- **#4 `10012374` — Depressed mood**
  - Evidence: `Depression`
  - Rationale (deepseek_validated): Depressed mood is a symptom of depression, but the benchmark phrase 'Depression' can refer to a broader condition. The evidence includes 'depressed mood' and 'depressive disorder', but the match is less direct than 'Sad mood'.

### VOLTAREN-XR.3.ann_TT3

**Real phrase:** `chest discomfort`  
**Gold:** `10008469` · **Baseline top1:** `10053155` · **DeepSeek top1:** `10008469` · **Route:** `TOP_K_HUMAN_CHOICE`

- **#3 `10053155` — Gastric irritation**
  - Evidence: `chest discomfort`
  - Rationale (deepseek_validated): The term 'Gastric irritation' includes 'epigastric discomfort' as a synonym, which is a form of discomfort but located in the epigastric region, not specifically the chest; the benchmark phrase 'chest discomfort' is broader and not necessarily gastrointestinal.
- **#2 `10008479` — Retrosternal pain**
  - Evidence: `chest discomfort`
  - Rationale (deepseek_validated): The term 'Retrosternal pain' is a type of chest pain, which is related to 'chest discomfort' but not identical; the benchmark phrase emphasizes discomfort rather than pain, and the allowed source evidence only contains 'chest discomfort', not 'chest pain'.
- **#4 `10024968` — chest infections**
  - Evidence: `chest discomfort`
  - Rationale (deepseek_validated): The term 'chest infections' refers to an infectious condition, which is distinct from the symptom 'chest discomfort'; the allowed source evidence only contains 'chest discomfort', not 'chest infection'.
- **#1 `10008469` — Chest discomfort**
  - Evidence: `chest discomfort`
  - Rationale (deepseek_validated): The term 'Chest discomfort' is an exact lexical match to the benchmark phrase 'chest discomfort', and the terminology support includes 'chest discomfort' as a synonym, confirming direct alignment.
- **#5 `10019030` — Hair discoloration**
  - Evidence: `chest discomfort`
  - Rationale (deepseek_validated): The term 'Hair discoloration' is completely unrelated to 'chest discomfort'; there is no semantic or clinical connection, and the allowed source evidence does not support this mapping.

### LIPITOR.773.ann_TT2

**Real phrase:** `extremly bad pains in arms`  
**Gold:** `10033371` · **Baseline top1:** `10033425` · **DeepSeek top1:** `10033425` · **Route:** `TOP_K_HUMAN_CHOICE`

- **#1 `10033425` — Pain in limb**
  - Evidence: `extremly bad pains in arms`
  - Rationale (deepseek_validated): The phrase 'extremly bad pains in arms' directly matches the concept of pain in the arms, which is a type of limb pain. The terminology support includes 'arms pain' as a synonym, and the historical support shows a match with 'arms pain'.
- **#2 `10028391` — Musculoskeletal Pain**
  - Evidence: `extremly bad pains in arms`
  - Rationale (deepseek_validated): The phrase mentions 'pains in arms', which could be part of musculoskeletal pain. The terminology support includes 'shoulders and arms pain' as a synonym, indicating some relevance, but 'Musculoskeletal Pain' is broader and less specific than 'Pain in limb'.
- **#3 `10033371` — Acute onset pain**
  - Evidence: `extremly bad pains in arms`
  - Rationale (deepseek_validated): The phrase describes 'extremly bad pains', which aligns with the general concept of pain, but 'Acute onset pain' emphasizes sudden onset, which is not specified in the phrase. The terminology support includes 'pain' and 'worst pain', but the code is less specific to the location (arms).
- **#4 `10000087` — Abdominal pain upper**
  - Evidence: `extremly bad pains in arms`
  - Rationale (deepseek_validated): The phrase specifies 'pains in arms', which is not related to the abdomen. 'Abdominal pain upper' is a mismatch because the location is clearly the arms, not the stomach or upper abdomen.
- **#5 `10022998` — Irritable Mood**
  - Evidence: `extremly bad pains in arms`
  - Rationale (deepseek_validated): The phrase describes physical pain in the arms, not a mood state. 'Irritable Mood' is unrelated to pain and is a mismatch, as there is no indication of irritability or mood disturbance.

## Data/source note

MedNorm official source/licence authority: DOI `10.17632/b9x7xxb9sz.1`, CC BY-NC 3.0. The Hugging Face copy is used only as a transport convenience and does not override the official licence.