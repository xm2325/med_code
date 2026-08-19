from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable

from .clinical_context import classify_assertion


DEMO_TERMINOLOGY_SOURCE = {
    "name": "NCI CTCAE v6.0 clean copy (MedDRA 28.0)",
    "url": "https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events",
    "level": "LLT",
    "use": "Synthetic, non-clinical MedCode demonstration subset",
}


@dataclass(frozen=True)
class ADRTerm:
    code: str
    term: str
    soc: str
    aliases: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


# The codes and English terms below are a small, explicitly versioned subset of the
# public NCI CTCAE v6.0 clean-copy spreadsheet. They are not a replacement for an
# authorised full MedDRA distribution.
DEMO_ADR_TERMS: tuple[ADRTerm, ...] = (
    ADRTerm("10028813", "Nausea", "Gastrointestinal disorders", ("nausea", "nauseated", "feeling sick", "恶心", "反胃")),
    ADRTerm("10047700", "Vomiting", "Gastrointestinal disorders", ("vomiting", "vomited", "emesis", "being sick", "呕吐")),
    ADRTerm("10012727", "Diarrhea", "Gastrointestinal disorders", ("diarrhea", "diarrhoea", "loose stools", "watery stools", "腹泻", "稀便")),
    ADRTerm("10028130", "Mucositis oral", "Gastrointestinal disorders", ("oral mucositis", "mouth ulcers", "oral ulcers", "mouth ulceration", "stomatitis", "口腔溃疡", "口腔黏膜炎")),
    ADRTerm("10055356", "Upper gastrointestinal hemorrhage", "Gastrointestinal disorders", ("upper gastrointestinal hemorrhage", "upper gi bleed", "gastrointestinal bleeding", "haematemesis", "hematemesis", "melena", "黑便", "呕血", "上消化道出血")),
    ADRTerm("10001551", "Alanine aminotransferase increased", "Investigations", ("alanine aminotransferase increased", "alt increased", "raised alt", "elevated alt", "alt升高", "丙氨酸转氨酶升高")),
    ADRTerm("10003481", "Aspartate aminotransferase increased", "Investigations", ("aspartate aminotransferase increased", "ast increased", "raised ast", "elevated ast", "ast升高", "天门冬氨酸转氨酶升高")),
    ADRTerm("10049182", "White blood cell decreased", "Investigations", ("white blood cell decreased", "low white count", "wbc decreased", "leukopenia", "leucopenia", "白细胞减少", "白细胞低")),
    ADRTerm("10029366", "Neutrophil count decreased", "Investigations", ("neutrophil count decreased", "low neutrophils", "neutropenia", "中性粒细胞减少", "中性粒细胞低")),
    ADRTerm("10043554", "Thrombocytopenia", "Blood and lymphatic system disorders", ("thrombocytopenia", "low platelets", "platelet count decreased", "血小板减少", "血小板低")),
    ADRTerm("10002272", "Anemia", "Blood and lymphatic system disorders", ("anemia", "anaemia", "low hemoglobin", "low haemoglobin", "贫血", "血红蛋白降低")),
    ADRTerm("10001760", "Alopecia", "Skin and subcutaneous tissue disorders", ("alopecia", "hair loss", "hair thinning", "脱发", "头发变稀")),
    ADRTerm("10037868", "Rash maculo-papular", "Skin and subcutaneous tissue disorders", ("maculopapular rash", "maculo-papular rash", "red spotted rash", "斑丘疹", "红色斑丘疹")),
    ADRTerm("10037087", "Pruritus", "Skin and subcutaneous tissue disorders", ("pruritus", "itching", "itchy skin", "瘙痒", "皮肤发痒")),
    ADRTerm("10046735", "Urticaria", "Skin and subcutaneous tissue disorders", ("urticaria", "hives", "raised itchy welts", "荨麻疹", "风团")),
    ADRTerm("10016256", "Fatigue", "General disorders and administration site conditions", ("fatigue", "fatigued", "marked tiredness", "unusual tiredness", "乏力", "疲劳")),
    ADRTerm("10022095", "Injection site reaction", "General disorders and administration site conditions", ("injection site reaction", "injection-site reaction", "injection site redness", "injection site swelling", "注射部位反应", "注射处红肿")),
    ADRTerm("10051792", "Infusion related reaction", "Injury, poisoning and procedural complications", ("infusion related reaction", "infusion reaction", "reaction during infusion", "输液反应", "输注反应")),
    ADRTerm("10019211", "Headache", "Nervous system disorders", ("headache", "head pain", "头痛")),
    ADRTerm("10035664", "Pneumonia", "Infections and infestations", ("pneumonia", "chest infection", "肺炎", "肺部感染")),
    ADRTerm("10040555", "Shingles", "Infections and infestations", ("shingles", "herpes zoster", "zoster", "带状疱疹")),
    ADRTerm("10035742", "Pneumonitis", "Respiratory, thoracic and mediastinal disorders", ("pneumonitis", "interstitial pneumonitis", "drug induced pneumonitis", "药物性肺炎", "间质性肺炎")),
    ADRTerm("10069339", "Acute kidney injury", "Renal and urinary disorders", ("acute kidney injury", "acute renal injury", "aki", "急性肾损伤", "急性肾功能损伤")),
    ADRTerm("10011368", "Creatinine increased", "Investigations", ("creatinine increased", "raised creatinine", "elevated creatinine", "肌酐升高")),
    ADRTerm("10020639", "Hyperglycemia", "Metabolism and nutrition disorders", ("hyperglycemia", "hyperglycaemia", "high blood glucose", "血糖升高", "高血糖")),
    ADRTerm("10031282", "Osteoporosis", "Musculoskeletal and connective tissue disorders", ("osteoporosis", "骨质疏松")),
    ADRTerm("10038923", "Retinopathy", "Eye disorders", ("retinopathy", "retinal toxicity", "视网膜病变", "视网膜毒性")),
    ADRTerm("10047516", "Vision decreased", "Eye disorders", ("vision decreased", "reduced vision", "visual acuity decreased", "视力下降", "视力减退")),
    ADRTerm("10066899", "Venous thromboembolism", "Vascular disorders", ("venous thromboembolism", "venous thrombosis", "deep vein thrombosis", "dvt", "pulmonary embolism", "vte", "静脉血栓栓塞", "深静脉血栓", "肺栓塞")),
    ADRTerm("10020772", "Hypertension", "Vascular disorders", ("hypertension", "high blood pressure", "blood pressure increased", "高血压", "血压升高")),
    ADRTerm("10002218", "Anaphylaxis", "Immune system disorders", ("anaphylaxis", "anaphylactic reaction", "过敏性休克", "严重过敏反应")),
    ADRTerm("10046300", "Upper respiratory infection", "Infections and infestations", ("upper respiratory infection", "upper respiratory tract infection", "urti", "上呼吸道感染")),
)


RA_DRUG_ALIASES: dict[str, tuple[str, ...]] = {
    "Methotrexate": ("methotrexate", "mtx", "甲氨蝶呤"),
    "Leflunomide": ("leflunomide", "来氟米特"),
    "Hydroxychloroquine": ("hydroxychloroquine", "hcq", "羟氯喹"),
    "Sulfasalazine": ("sulfasalazine", "ssz", "柳氮磺吡啶"),
    "Adalimumab": ("adalimumab", "阿达木单抗"),
    "Etanercept": ("etanercept", "依那西普"),
    "Infliximab": ("infliximab", "英夫利昔单抗"),
    "Tocilizumab": ("tocilizumab", "托珠单抗"),
    "Rituximab": ("rituximab", "利妥昔单抗"),
    "Tofacitinib": ("tofacitinib", "托法替布"),
    "Upadacitinib": ("upadacitinib", "乌帕替尼"),
    "Prednisolone": ("prednisolone", "泼尼松龙", "泼尼松"),
    "Naproxen": ("naproxen", "萘普生"),
}


_ZH_NEGATION = re.compile(r"(?:无|否认|未见|没有|排除)(?:明显)?")
_ZH_UNCERTAINTY = re.compile(r"(?:疑似|考虑|可能|不排除|待排)")
_ZH_HISTORICAL = re.compile(r"(?:既往|曾有|已缓解|已恢复)")


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias)
    if re.search(r"[A-Za-z0-9]", alias):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.I)
    return re.compile(escaped, re.I)


def _sentence_context(text: str, start: int, end: int) -> str:
    left_candidates = [text.rfind(mark, 0, start) for mark in (".", ";", "。", "；", "\n")]
    left = max(left_candidates) + 1
    right_candidates = [text.find(mark, end) for mark in (".", ";", "。", "；", "\n")]
    right_values = [value for value in right_candidates if value >= 0]
    right = min(right_values) + 1 if right_values else len(text)
    return text[left:right].strip()


def _classify_context(context: str) -> str:
    if _ZH_NEGATION.search(context):
        return "negated"
    if _ZH_UNCERTAINTY.search(context):
        return "uncertain"
    if _ZH_HISTORICAL.search(context):
        return "historical_or_resolved"
    return classify_assertion(context)


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", value.casefold()))


def _fuzzy_alias(text: str, alias: str) -> tuple[float, str, int, int]:
    target = _normalise(alias)
    if not target:
        return 0.0, "", 0, 0
    words = list(re.finditer(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text, re.I))
    if not words:
        return 0.0, "", 0, 0
    target_words = target.split()
    window_sizes = {max(1, len(target_words) - 1), len(target_words), len(target_words) + 1}
    best = (0.0, "", 0, 0)
    for size in window_sizes:
        for index in range(0, max(1, len(words) - size + 1)):
            chunk = words[index : index + size]
            if not chunk:
                continue
            start, end = chunk[0].start(), chunk[-1].end()
            quote = text[start:end]
            score = SequenceMatcher(None, _normalise(quote), target).ratio()
            if score > best[0]:
                best = (score, quote, start, end)
    return best


class OneLineADRCoder:
    """Evidence-first mapper for short RA treatment adverse-event records.

    This deliberately small, deterministic mapper powers the synthetic demo. A production
    deployment should fit the repository's validated candidate generator against an
    authorised full MedDRA terminology and locally approved clinical data.
    """

    def __init__(self, terms: Iterable[ADRTerm] = DEMO_ADR_TERMS) -> None:
        self.terms = tuple(terms)
        if not self.terms:
            raise ValueError("terms must contain at least one MedDRA concept")

    @staticmethod
    def _drug_mentions(text: str) -> list[str]:
        found: list[str] = []
        for drug, aliases in RA_DRUG_ALIASES.items():
            if any(_alias_pattern(alias).search(text) for alias in aliases):
                found.append(drug)
        return found

    def _exact_matches(self, text: str) -> list[dict]:
        matches: list[dict] = []
        for concept in self.terms:
            hits: list[tuple[int, int, str, str]] = []
            for alias in sorted(concept.aliases, key=len, reverse=True):
                for hit in _alias_pattern(alias).finditer(text):
                    hits.append((hit.start(), hit.end(), hit.group(0), alias))
            if not hits:
                continue
            start, end, quote, alias = sorted(hits, key=lambda row: (-(row[1] - row[0]), row[0]))[0]
            context = _sentence_context(text, start, end)
            assertion = _classify_context(context)
            base = 0.98 if alias.casefold() == concept.term.casefold() else (0.94 if len(alias) >= 8 else 0.90)
            score = {
                "affirmed": base,
                "uncertain": min(base, 0.74),
                "historical_or_resolved": min(base, 0.68),
                "family_history": min(base, 0.28),
                "negated": min(base, 0.12),
            }.get(assertion, min(base, 0.55))
            matches.append({
                "code": concept.code,
                "term": concept.term,
                "soc": concept.soc,
                "score": round(score, 3),
                "matched_alias": alias,
                "evidence": {"quote": quote, "start": start, "end": end, "context": context},
                "assertion": assertion,
                "match_type": "exact_alias",
            })
        rank = {"affirmed": 0, "uncertain": 1, "historical_or_resolved": 2, "family_history": 3, "negated": 4}
        return sorted(matches, key=lambda row: (rank.get(row["assertion"], 9), -row["score"], row["evidence"]["start"]))

    def _fuzzy_candidates(self, text: str, limit: int = 3) -> list[dict]:
        candidates: list[dict] = []
        for concept in self.terms:
            best = (0.0, "", 0, 0, "")
            for alias in concept.aliases:
                score, quote, start, end = _fuzzy_alias(text, alias)
                if score > best[0]:
                    best = (score, quote, start, end, alias)
            if best[0] >= 0.76:
                context = _sentence_context(text, best[2], best[3])
                candidates.append({
                    "code": concept.code,
                    "term": concept.term,
                    "soc": concept.soc,
                    "score": round(best[0] * 0.72, 3),
                    "matched_alias": best[4],
                    "evidence": {"quote": best[1], "start": best[2], "end": best[3], "context": context},
                    "assertion": _classify_context(context),
                    "match_type": "fuzzy_alias",
                })
        return sorted(candidates, key=lambda row: -row["score"])[:limit]

    def map_line(self, text: str, *, record_id: str | None = None) -> dict:
        source = str(text or "").strip()
        if not source:
            raise ValueError("text must contain one non-empty clinical record")

        candidates = self._exact_matches(source)
        if not candidates:
            candidates = self._fuzzy_candidates(source)

        affirmed = [row for row in candidates if row["assertion"] == "affirmed"]
        reviewable = [row for row in candidates if row["assertion"] in {"uncertain", "historical_or_resolved"}]
        primary: dict | None = None
        reasons: list[str] = []

        if affirmed:
            primary = affirmed[0]
            if len(affirmed) == 1 and primary["match_type"] == "exact_alias" and primary["score"] >= 0.88:
                route = "AUTO_CANDIDATE"
            else:
                route = "TOP_K_HUMAN_CHOICE"
                reasons.append("The line contains multiple affirmed events or only an approximate terminology match.")
        elif reviewable:
            primary = reviewable[0]
            route = "TOP_K_HUMAN_CHOICE"
            reasons.append("The event is uncertain or historical/resolved and requires clinical confirmation.")
        else:
            route = "FULL_EXPERT_REVIEW"
            if candidates:
                reasons.append("Only negated, family-history, or unsupported event text was found.")
            else:
                reasons.append("No concept in the limited demo terminology matched this record.")

        if not self._drug_mentions(source):
            reasons.append("No recognised RA treatment was found; attribution cannot be inferred.")

        confidence = float(primary["score"]) if primary else 0.0
        encoded_events = [row for row in candidates if row["assertion"] in {"affirmed", "uncertain", "historical_or_resolved"}]
        return {
            "record_id": record_id,
            "text": source,
            "drug_mentions": self._drug_mentions(source),
            "route": route,
            "confidence": round(confidence, 3),
            "primary": primary,
            "encoded_events": encoded_events,
            "candidates": candidates[:5],
            "review_reasons": reasons,
            "terminology": dict(DEMO_TERMINOLOGY_SOURCE),
            "synthetic_demo_only": True,
        }

    def map_lines(self, texts: Iterable[str]) -> list[dict]:
        return [self.map_line(text, record_id=f"ADR-{index:03d}") for index, text in enumerate(texts, start=1)]


def demo_terminology() -> list[dict]:
    return [term.to_dict() for term in DEMO_ADR_TERMS]
