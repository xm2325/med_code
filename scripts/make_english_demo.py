from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "demo" / "ra_adr_meddra_demo.html"
OUTPUT = ROOT / "demo" / "ra_adr_meddra_demo.en.html"


# These are the five synthetic source lines that were originally written in Chinese.
# The English page keeps the same record ids, expected LLTs and evidence semantics.
RECORD_TRANSLATIONS = {
    "类风湿关节炎复诊：甲氨蝶呤加量后出现口腔黏膜炎。":
        "RA follow-up: oral mucositis appeared after a methotrexate dose increase.",
    "类风湿患者服用来氟米特后诉脱发明显。":
        "The RA patient reported marked hair loss after taking leflunomide.",
    "类风湿关节炎用柳氮磺吡啶两周后发现血小板减少。":
        "Platelet count decreased after two weeks of sulfasalazine for RA.",
    "依那西普治疗类风湿后，注射处红肿持续两天。":
        "Injection-site redness and swelling lasted two days after etanercept for RA.",
    "托珠单抗输注过程中出现输液反应，伴寒战和面部潮红。":
        "An infusion-related reaction with chills and facial flushing occurred during tocilizumab infusion.",
}


TERM_TRANSLATIONS = {
    "恶心": "nausea", "反胃": "feeling sick", "呕吐": "vomiting",
    "腹泻": "diarrhea", "稀便": "loose stools", "口腔溃疡": "oral ulcers",
    "口腔黏膜炎": "oral mucositis", "黑便": "melena", "呕血": "hematemesis",
    "上消化道出血": "upper gastrointestinal hemorrhage", "alt升高": "ALT increased",
    "丙氨酸转氨酶升高": "alanine aminotransferase increased", "ast升高": "AST increased",
    "天门冬氨酸转氨酶升高": "aspartate aminotransferase increased",
    "白细胞减少": "white blood cell decreased", "白细胞低": "low white count",
    "中性粒细胞减少": "neutrophil count decreased", "中性粒细胞低": "low neutrophils",
    "血小板减少": "thrombocytopenia", "血小板低": "low platelets",
    "贫血": "anemia", "血红蛋白降低": "low hemoglobin", "脱发": "hair loss",
    "头发变稀": "hair thinning", "斑丘疹": "maculopapular rash",
    "红色斑丘疹": "red spotted rash", "瘙痒": "itching", "皮肤发痒": "itchy skin",
    "荨麻疹": "urticaria", "风团": "raised itchy welts", "乏力": "fatigue",
    "疲劳": "fatigued", "注射部位反应": "injection site reaction",
    "注射处红肿": "injection site redness", "输液反应": "infusion reaction",
    "输注反应": "reaction during infusion", "头痛": "headache", "肺炎": "pneumonia",
    "肺部感染": "chest infection", "带状疱疹": "shingles", "药物性肺炎": "drug induced pneumonitis",
    "间质性肺炎": "interstitial pneumonitis", "急性肾损伤": "acute kidney injury",
    "急性肾功能损伤": "acute renal injury", "肌酐升高": "raised creatinine",
    "血糖升高": "high blood glucose", "高血糖": "hyperglycemia", "骨质疏松": "osteoporosis",
    "视网膜病变": "retinopathy", "视网膜毒性": "retinal toxicity",
    "视力下降": "vision decreased", "视力减退": "reduced vision",
    "静脉血栓栓塞": "venous thromboembolism", "深静脉血栓": "deep vein thrombosis",
    "肺栓塞": "pulmonary embolism", "高血压": "hypertension", "血压升高": "blood pressure increased",
    "过敏性休克": "anaphylaxis", "严重过敏反应": "anaphylactic reaction",
    "上呼吸道感染": "upper respiratory infection",
    "甲氨蝶呤": "methotrexate", "来氟米特": "leflunomide", "羟氯喹": "hydroxychloroquine",
    "柳氮磺吡啶": "sulfasalazine", "阿达木单抗": "adalimumab", "依那西普": "etanercept",
    "英夫利昔单抗": "infliximab", "托珠单抗": "tocilizumab", "利妥昔单抗": "rituximab",
    "托法替布": "tofacitinib", "乌帕替尼": "upadacitinib", "泼尼松龙": "prednisolone",
    "泼尼松": "prednisone", "萘普生": "naproxen",
}


# Visible labels and copy in the page and in its deterministic JavaScript renderer.
UI_TRANSLATIONS = {
    'lang="zh-CN"': 'lang="en"',
    "治疗相关不良反应一行记录到": "drug-related adverse-reaction line to",
    "代码的合成交互演示": "code: synthetic interactive demo",
    "不良反应编码台": "Adverse-reaction coding desk",
    "合成演示": "Synthetic demo",
    "子集": "subset",
    "把一行不良反应记录，变成可复核的": "Turn one adverse-reaction line into a reviewable",
    "代码。": "code.",
    "面向": "For",
    "治疗安全监测：识别药物与事件，保留原文证据，标记否定和不确定语境，并把需要判断的案例送回医护人员。": "RA treatment safety monitoring: identify drugs and events, preserve verbatim evidence, flag negated and uncertain context, and route cases needing judgment back to clinical staff.",
    "编码流程": "Coding flow",
    "解析原文": "Parse the record",
    "一行医护记录": "One clinician line",
    "用药": "Treatment",
    "锁定证据": "Lock evidence",
    "逐字片段、位置与语境": "Verbatim span, offset, and context",
    "编码与反馈": "Coding and feedback",
    "候选": "Candidates",
    "人工闭环": "Human-in-the-loop",
    "非临床使用：": "Non-clinical use:",
    "下方": "The",
    "条记录均为合成数据；代码来自": "records below are synthetic; codes come from",
    "的": "of",
    "演示子集，不代表因果判断或完整": "demo subset; this is not a causality assessment or a complete",
    "许可词表。": "licensed dictionary.",
    "查看术语来源": "View terminology source",
    "试编码一行记录": "Code a single record",
    "从": "Choose from",
    "条内嵌合成记录中选择": "embedded synthetic records",
    "选择合成记录": "Select a synthetic record",
    "自定义输入": "Custom input",
    "随机一条": "Random record",
    "一行不良反应记录": "One-line adverse-reaction record",
    "例如：": "Example:",
    "生成编码建议": "Generate coding suggestion",
    "全部": "All",
    "条记录与术语均已内嵌在此": "records and terminology are embedded in this",
    "；支持离线选择、随机抽取和中英文自定义输入。": "; supports offline selection, random sampling, and custom input in English or Chinese.",
    "等待一行记录": "Waiting for a record",
    "结果会在这里显示代码、术语、证据和分流建议。": "Results will show the code, term, evidence, and routing recommendation here.",
    "我用了什么方法，把记录变成": "Which method turns a record into",
    "同一条记录切换三种可解释路径。当前": "Switch the same record across three explainable paths. The",
    "实际运行的是证据优先基线；另外两种是可落地的": "runnable path is the evidence-first baseline; the other two are implementable",
    "方案示意，用来说明": "design patterns showing how",
    "、检索和人工闭环应如何组合。": ", retrieval, and human review should work together.",
    "编码方法选择": "Choose a coding method",
    "当前演示": "Live demo",
    "证据优先基线": "Evidence-first baseline",
    "推荐路径": "Recommended path",
    "当前可运行": "Runnable locally",
    "这条记录会产出什么？": "What does this record produce?",
    "自动候选": "Auto candidate",
    "未匹配": "No match",
    "演示概况": "Demo overview",
    "首批合成记录": "Synthetic records",
    "需人工确认": "Human confirmation",
    "本机已记录反馈": "Feedback saved locally",
    "不良反应工作列": "Adverse-reaction worklist",
    "点击“审核”查看原始证据并提交接受、改码或无代码反馈。反馈仅保存在当前浏览器，可导出": "Click “Review” to inspect source evidence and submit accept, recode, or no-code feedback. Feedback stays in this browser and can be exported as",
    "记录筛选": "Filter records",
    "搜索记录、药物、术语或代码…": "Search records, drugs, terms, or codes…",
    "搜索记录": "Search records",
    "按药物筛选": "Filter by drug",
    "全部药物": "All drugs",
    "按分流筛选": "Filter by route",
    "全部分流": "All routes",
    "人工选择": "Human choice",
    "完整专家复核": "Full expert review",
    "按反馈状态筛选": "Filter by feedback status",
    "全部反馈状态": "All feedback states",
    "待反馈": "Awaiting feedback",
    "已反馈": "Feedback recorded",
    "导出反馈": "Export feedback",
    "一行记录": "One-line record",
    "证据": "Evidence",
    "术语": "Term",
    "分流": "Route",
    "反馈": "Feedback",
    "审核编码建议": "Review coding suggestion",
    "关闭审核": "Close review",
    "术语版本：": "Terminology version:",
    "演示子集": "Demo subset",
    "实际执行的确定性流程：同义词": "Actual deterministic flow: synonym",
    "别名命中": "alias match",
    "语境断言": "context assertion",
    "证据定位": "evidence localization",
    "分流。它不调用外部": "routing. It does not call an external",
    "，结果可离线重放。": "; results can be replayed offline.",
    "输入与识别": "Input and recognition",
    "保留原文，识别": "Preserve the source line; identify",
    "药物与事件别名。": "drug and event aliases.",
    "语境检查": "Context check",
    "在原文附近识别肯定、否定、不确定或既往语境。": "Identify affirmed, negated, uncertain, or historical context near the mention.",
    "代码与分流": "Coding and routing",
    "从版本化": "From the versioned",
    "子集中给出候选，并决定自动、": "subset, propose candidates, and decide between auto,",
    "或专家复核。": "Top-K choice, or expert review.",
    "真实运行结果：可复核、可离线；限制是": "Actual runnable result: reviewable and offline; limitation:",
    "词表很小，不能代表完整": "the vocabulary is small and cannot represent a complete",
    "或临床准确率。": "MedDRA release or clinical accuracy.",
    "受限": "Constrained",
    "方案示意": "design pattern",
    "把固定的": "Put the fixed",
    "和编码约束直接放进": "and coding constraints directly into",
    "让": "let",
    "，让": ", so that",
    "做抽取与选择。它适合做语言理解基线，但不能让模型凭记忆生成": "performs extraction and selection. It is useful as a language-understanding baseline, but the model must not generate",
    "结构化抽取": "Structured extraction",
    "提取": "extract",
    "和原文": "and source",
    "封闭集合选择": "Closed-set selection",
    "只允许从": "Allow selection only from",
    "中的术语": "the terms in",
    "选择，或返回": "or return",
    "机器校验": "Machine validation",
    "校验": "Validate",
    "、代码集合、": ", the code set,",
    "与": "and",
    "；失败即人工复核。": "; failures go to human review.",
    "这是可执行的": "This is an executable",
    "契约示意；本地": "contract example; the local",
    "没有真的调用": "does not actually call",
    "，避免把假设输出误当成模型实测。": ", avoiding presenting hypothetical output as measured model behavior.",
    "冻结候选重排": "Frozen-candidate reranking",
    "推荐架构": "Recommended architecture",
    "先从机构授权、固定版本的": "First retrieve from an institution-authorized, version-pinned",
    "索引检索候选，再让": "index, then let",
    "只在冻结候选集内重排与解释；代码集合不能被模型增删。": "rerank and explain only within the frozen candidate set; the model cannot add or remove codes.",
    "事件与语境": "Event and context",
    "先切分事件": "Split event",
    "，标记否定": ", flag negation,",
    "不确定": "uncertainty,",
    "既往并保留证据": "historical status, and retain evidence",
    "版本化检索": "Versioned retrieval",
    "检索": "Retrieve",
    "、同义词、层级和来源，形成可审计": ", synonyms, hierarchy, and provenance to form an auditable",
    "受限重排": "Constrained reranking",
    "只能重排": "may only rerank",
    "；校验后按证据和不确定性路由人工闭环。": "; after validation, route by evidence and uncertainty to human review.",
    "生产推荐路径：": "Production recommendation:",
    "提供术语证据，": " provides terminology evidence,",
    "只做受限判断；正式使用仍需授权词表、脱敏、版本锁定和受训人员复核。": " performs constrained decisions; production still requires an authorized dictionary, de-identification, version pinning, and trained review.",
    "无": "no",
    "否认": "denies",
    "未见": "no evidence of",
    "没有": "no",
    "排除": "ruled out",
    "疑似": "suspected",
    "考虑": "consider",
    "可能": "possible",
    "不排除": "cannot exclude",
    "待排": "to be ruled out",
    "既往": "historical",
    "曾有": "previously had",
    "已缓解": "resolved",
    "已恢复": "recovered",
    "同一行包含多个肯定事件，需要确认主编码。": "The line contains multiple affirmed events; confirm the primary code.",
    "事件语境为不确定或既往，需要医护人员确认。": "The event is uncertain or historical; clinician confirmation is required.",
    "仅发现否定或不支持的事件文本。": "Only negated or unsupported event text was found.",
    "演示词表内未找到匹配术语。": "No matching term was found in the demo dictionary.",
    "未识别到": "No recognised",
    "治疗药物，不能推断归因。": "RA treatment was identified; attribution cannot be inferred.",
    "匹配": "Match",
    "未生成自动代码": "No automatic code",
    "匹配置信度": "Match confidence",
    "需要澄清": "Needs clarification",
    "需校验": "Needs validation",
    "冻结": "Frozen",
    "没有额外检索层：术语必须随": "No additional retrieval layer: terminology must be supplied with",
    "一起进入上下文，代码不得来自模型记忆。": "in the context; codes must not come from model memory.",
    "：代码来自本地": ": code from local",
    "命中；原文": "match; source text",
    "、字符": ", character",
    "和": "and",
    "会随结果保存。": "are retained with the result.",
    "实际执行：确定性本地编码器": "Executed: deterministic local coder",
    "证据可重放": "Evidence replayable",
    "因果性未评估": "Causality not assessed",
    "方案输出：": "Design output: ",
    "失败则人工复核": "failures route to human review",
    "冻结候选集": "Frozen candidate set",
    "只重排不生成代码": "Rerank only; no code generation",
    "没有符合当前筛选条件的记录。": "No records match the current filters.",
    "载入": "Load",
    "审核": "Review",
    "已接受": "Accepted",
    "已改码": "Recoded",
    "无代码": "No code",
    "已升级": "Escalated",
    "当前建议": "Current suggestion",
    "接受当前建议": "Accept current suggestion",
    "选择其他": "Choose another",
    "代码": "code",
    "该行不应编码": "This line should not be coded",
    "升级完整专家复核": "Escalate to full expert review",
    "替代代码（改码时使用）": "Alternative code (for recoding)",
    "请选择…": "Select…",
    "反馈说明": "Feedback note",
    "记录接受理由、纠错依据或需要进一步核实的信息…": "Record acceptance rationale, correction basis, or information needing verification…",
    "保存反馈": "Save feedback",
    "清除本条": "Clear this item",
    "请选择替代": "Select an alternative",
    "反馈已保存在当前浏览器": "Feedback saved in this browser",
    "本条反馈已清除": "Feedback cleared for this item",
    "请先输入一行记录": "Enter a record first",
}


def translate_payload(payload: object) -> object:
    def walk(value: object) -> object:
        if isinstance(value, str):
            for old, new in sorted(RECORD_TRANSLATIONS.items(), key=lambda item: -len(item[0])):
                value = value.replace(old, new)
            for old, new in sorted(TERM_TRANSLATIONS.items(), key=lambda item: -len(item[0])):
                value = value.replace(old, new)
            return value
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value

    translated = walk(payload)
    if isinstance(translated, dict):
        for term in translated.get("terms", []):
            if isinstance(term, dict) and isinstance(term.get("aliases"), list):
                term["aliases"] = list(dict.fromkeys(term["aliases"]))
        if isinstance(translated.get("drugAliases"), dict):
            translated["drugAliases"] = {
                drug: list(dict.fromkeys(aliases)) if isinstance(aliases, list) else aliases
                for drug, aliases in translated["drugAliases"].items()
            }
    return translated


def main() -> None:
    html = SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="medcodePayload">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError("embedded payload not found")
    payload = translate_payload(json.loads(match.group(1)))
    html = html[: match.start(1)] + "__ENGLISH_PAYLOAD__" + html[match.end(1) :]
    for old, new in sorted(UI_TRANSLATIONS.items(), key=lambda item: -len(item[0])):
        html = html.replace(old, new)
    cleanups = {
        "Which method turns a record into MedDRA code?": "Which method turns a record into a MedDRA code?",
        "The Demo runnable path": "The runnable Demo path",
        "Demo Actual deterministic flow": "Demo actual deterministic flow",
        "auto,Top-K Top-K choice": "auto, Top-K choice",
        "limitation: Demo the vocabulary": "limitation: the Demo vocabulary",
        "complete MedDRA MedDRA release": "complete MedDRA release",
        "directly into prompt, so that LLM": "directly into the prompt, so that the LLM",
        "drug、event、assertion and source span": "drug, event, assertion, and source span",
        "prompt the terms in allowlist": "the terms in the prompt allowlist",
        "code set,quote and offset": "code set, quote and offset",
        "negation,/uncertainty,/historical status": "negation, uncertainty, or historical status",
        "Retrieval LLT/PT": "Retrieve LLT/PT",
        "Production recommendation:RAG  provides terminology evidence,LLM  performs": "Production recommendation: RAG provides terminology evidence; the LLM performs",
        "No recognised RA RA treatment": "No recognised RA treatment",
        "Terminology version:NCI": "Terminology version: NCI",
        "let LLM extract": "Let the LLM extract",
        "then let LLM rerank": "then let the LLM rerank",
        "Prompt-only No additional retrieval layer": "Prompt-only: no additional retrieval layer",
        "span Validate": "span validation",
    }
    for old, new in cleanups.items():
        html = html.replace(old, new)
    for old, new in {"？": "?", "。": ".", "、": ", ", "：": ": ", "；": "; ", "，": ", ", "…": "..."}.items():
        html = html.replace(old, new)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__ENGLISH_PAYLOAD__", encoded)
    html = html.replace("<title>MedCode · RA Adverse-reaction coding desk</title>", "<title>MedCode · RA adverse-reaction coding desk</title>")
    if re.search(r"[\u3400-\u9fff]", html):
        leftovers = sorted(set(re.findall(r"[^\x00-\x7f]*[\u3400-\u9fff][^\x00-\x7f]*", html)))
        raise RuntimeError(f"Chinese text remains: {leftovers[:20]}")
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Built {OUTPUT} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
