# RA ADR → MedDRA 演示：方法可视化建议

## 结论：当前页面应先诚实说明“实际运行的方法”

目前的独立 Demo **不是 LLM 或 RAG**。它运行的是一个可离线复现的、证据优先的术语匹配器：

1. 从一行记录识别 RA 药物别名；
2. 在 32 个演示 LLT 的同义词中做精确匹配；
3. 若无精确命中，再做受限的模糊字符串匹配；
4. 在事件所在句识别肯定、否定、不确定、既往/已缓解语境；
5. 给出 Top-K 候选，按规则分流到自动候选、人工选择或完整复核。

这比把它描述成“LLM 已完成医学编码”更可信，也正好能解释为什么结果可复核。LLM+RAG 应展示为可落地的生产架构和可复制的受限提示词，不能伪装成这个静态 HTML 已经调用了模型。

## 推荐的信息架构：一个记录，一条可展开的决策轨迹

将现有 Live coder 的输出区改为 `编码轨迹（Coding trace）`。上半区保留最终建议；下半区是横向步骤条。选择样本、随机一条或编辑一行文本后，步骤条同步更新。

```text
临床原文
   ↓
证据与语境抽取
   ↓
术语/RAG 候选检索（Top-K 冻结）
   ↓
可选 LLM 重排（只可排序，不可造码）
   ↓
路由与人工确认
```

### 第一屏：回答“结果是什么、凭什么”

放在结果区，默认始终可见：

| 字段 | 当前 Demo 可真实提供 | 目的 |
| --- | --- | --- |
| 建议 MedDRA LLT | `10028813 · Nausea` | 立即回答编码结果 |
| 系统器官分类 | `Gastrointestinal disorders` | 给编码者医学上下文 |
| 原文证据 | 高亮 `nausea`，显示字符偏移 | 证明没有脱离原文造结论 |
| 语境 | 肯定 / 否定 / 不确定 / 既往 | 避免把“否认恶心”编码成恶心 |
| 分流 | 自动候选 / Top-K 人工选择 / 完整复核 | 说明系统是否要求人看 |
| 非因果提示 | “药物被识别；不代表药物导致事件” | 避免混淆 normalisation 与因果评估 |

“匹配置信度”应改名为 **检索/匹配分数（非校准概率）**，并在旁边加信息提示。当前分数是别名命中与语境规则的诊断分数，不是临床风险或正确率概率。

### 第二屏：展开“这个结果是怎样来的”

使用 5 个可点击步骤卡；只展开当前卡，避免一次把技术细节压给临床用户。

1. **输入与结构化抽取**
   - 原始一行记录（不改写）
   - 识别到的 RA 药物：`Methotrexate`；药物字符片段/别名
   - 事件片段：`nausea`；`start=11, end=17`
   - assertion：`affirmed`；事件所在句
   - 没有药物、多个事件、或否定事件时的醒目告警

2. **候选生成：术语检索（Demo 实际运行）**
   - 顶部明确写：`当前运行：本地 LLT 同义词检索（32 条演示术语）`
   - 表格字段：rank、LLT code、term、matched synonym、match method、score、assertion
   - 对无命中输入展示：`未在演示词表中检到候选；不生成代码，送完整复核`。
   - 例：`mouth ulcers → Mucositis oral / 10028130 / exact alias / 0.94`。

3. **RAG：生产方案（展开为设计而非伪造运行记录）**
   - 标识：`生产架构 / 未在此离线 Demo 执行`
   - 可检索知识源卡片：授权的完整 MedDRA 版本（PT/LLT、同义词、层级、编码指南）；仅限 TRAIN 的历史专家编码记忆；可选的本地医学文本向量索引。
   - 每个返回项必须有：`source_type`、`terminology_version`、`retrieval_score`、`snippet`、`provenance_id`。
   - 关键约束：历史相似案例只能是 provenance，不是新病人的诊断证据；LLM 不接收完整未脱敏病历，默认只接收原文事件片段、语境和冻结候选。

4. **LLM 重排：受限 Prompt（生产可选）**
   - 标识：`可选 / 需要数据治理与评估`
   - 显示“复制 Prompt”按钮，代码块采用结构化 JSON 输入和 JSON schema 输出。
   - 强制字段：`prediction_id`、`ranked_codes`（必须与候选集完全相同）、`selected_code`、`uncertainty`、`evidence_quote`。
   - 校验条：`候选集是否被改变`、`代码是否在冻结候选中`、`证据是否为原文子串`。任一失败则回退至检索排序并路由人工复核。

5. **路由与人工闭环**
   - 显示具体触发原因而不是只显示标签：例如“一个肯定的精确别名命中；分数 ≥ 阈值”或“记录有 2 个肯定事件”。
   - 复核抽屉中保留原始候选集、模型/提示词版本、术语版本和时间戳；审核人可接受、换码、无代码、升级。
   - 导出时增加 `method`, `terminology_version`, `candidate_set_hash`, `prompt_version`（若 LLM 已使用）。

## 可直接放进 Demo 的 Prompt（只用于冻结候选重排）

不要让模型面对“自由生成 MedDRA code”的问题。以下内容适合展示为生产可选模块的输入模板：

```text
SYSTEM
You are a clinical terminology coding assistant. You rank an already frozen
candidate set for one adverse-event mention. You must not diagnose, assess
causality, add a code, remove a code, or use knowledge outside the supplied
record evidence and candidate terminology.

Return JSON only:
{
  "prediction_id": "...",
  "ranked_codes": ["..."],
  "selected_code": "...",
  "uncertainty": "low|medium|high",
  "evidence_quote": "exact substring from source_evidence",
  "rationale": "one short, grounded sentence"
}

USER
{
  "prediction_id": "SYN-001",
  "source_evidence": {"quote": "nausea", "assertion": "affirmed"},
  "clinical_context": "RA review: nausea started the morning after the weekly methotrexate dose.",
  "frozen_candidates": [
    {"code": "10028813", "term": "Nausea", "matched_synonym": "nausea", "source": "MedDRA 28.0 LLT"}
  ],
  "instruction": "Rank exactly this candidate set. Do not infer drug causality."
}
```

页面应在 Prompt 旁明确显示三条 guardrail：**候选在调用前冻结**、**JSON 校验失败即拒绝结果**、**LLM 输出不取代人工审核/因果评估**。

## 建议的三条代表性场景（优先于一张 50 条大表）

把选择器旁新增 `演示场景` 过滤器，帮助用户理解不同的算法行为：

| 场景 | 建议记录 | 要展示的效果 |
| --- | --- | --- |
| 清晰的单事件 | SYN-001（nausea） | 精确别名 → 一个候选 → 自动候选；展示证据和术语命中 |
| 不确定 / 需人工确认 | SYN-031、SYN-046、SYN-050 | 语境降低路由等级；仍显示候选，但不静默自动采纳 |
| 否定或多事件 | 自定义 `No anaphylaxis … later pneumonia`；或含 nausea + vomiting 的输入 | 否定候选置灰/排除；多肯定事件显示 Top-K 并要求确定主编码 |

原有 50 条表格仍可保留为“批量工作列”，但不要让它成为解释方法的主要视觉中心。

## 数据与代码接口：最小增量

现有 `OneLineADRCoder.map_line()` 已提供大部分演示所需字段：

- `drug_mentions`, `primary`, `candidates`, `encoded_events`；
- `evidence.quote/start/end/context`, `assertion`, `match_type`, `score`；
- `route`, `review_reasons`, `terminology`。

HTML 可由这些字段直接构造步骤 1、2、5。为把当前实运行与生产蓝图隔开，可在 JSON payload 添加固定元数据：

```json
{
  "method": {
    "active": "deterministic_lexical_fuzzy_demo",
    "active_label": "本地术语同义词检索 + 语境规则",
    "llm_reranking": {"available_in_production": true, "executed_in_demo": false},
    "rag": {"available_in_production": true, "executed_in_demo": false}
  }
}
```

对于真实生产实现，RAG 命中、冻结候选和 LLM 审计字段必须由服务端生成与持久化；不要把受许可的完整 MedDRA 或真实临床文本塞进离线静态页面。

## 验收标准

1. 用户在 10 秒内能读到：`当前 Demo 用的是本地术语匹配，不是已调用 LLM`。
2. 任意记录都能打开并查看原文证据、语境、候选列表与分流原因。
3. 自定义输入的否定、多事件、词表外三种情况都不产生误导性的“自动代码”。
4. LLM+RAG 区域明确标注为生产方案；能复制冻结候选 Prompt 与检查清单。
5. 审核导出包含所用方法和术语版本，保证之后可以复现决策。
