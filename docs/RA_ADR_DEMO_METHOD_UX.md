# RA ADR 一行记录到 MedDRA：真实实现与演示契约

## 当前实现

当页面由 FastAPI 服务时，同一条合成记录会真正调用 Roihu 上的
`Qwen/Qwen3.8-27B`，并独立运行三种方法。页面只把后端状态为 `LIVE`
的方法显示为模型结果；`ERROR`、`NOT_RUN` 或离线状态不会借用静态基线的
code，也不会生成伪造的 RAG evidence。

静态 HTML 仍然自包含 50 条合成记录和一个确定性演示匹配器，方便离线选择、
随机抽取和反馈演练。该匹配器明确标记为 `OFFLINE deterministic baseline`，
不冒充 Qwen 的返回。

```text
一行合成记录
  ├─ 方法 1：Qwen + 有界闭集 prompt ────────────────┐
  ├─ 方法 2：Qwen 事件抽取 → 确定性词法检索 ────────┼─ schema / span / code 校验 → 人工分流
  └─ 方法 3：Qwen 事件抽取 → Hybrid RAG Top-K 冻结
                              → Qwen 只重排 ─────────┘
```

三种方法的输出、错误和审计对象彼此独立。任何一种失败都不会污染另外两种。

## 方法 1：受限 Prompt-only

这是一条语言理解基线，不允许模型凭记忆自由生成 MedDRA code。

1. 服务端从本项目 RA 演示术语中构造有界 allowlist；在加载 850 行 CTCAE
   公开子集时也只把该小闭集放入 prompt，避免超过 8K context。
2. Qwen 返回 JSON：逐个事件的原文 `quote/start`、assertion，以及闭集 code
   或 `NO_CODE`。服务端用 `start + len(quote)` 产生 exclusive `end`，避免让
   LLM 做易错的机械字符计数。
3. 服务端逐项验证 JSON schema、逐字 `text[start:end]`、assertion 和 allowlist；
   返回中记录 `offset_source=SERVER_DERIVED_FROM_QUOTE_LENGTH`。
4. 模型添加不存在的 code、引用非原文片段或返回非法结构时，该事件被拒绝并送
   `FULL_EXPERT_REVIEW`，不会静默修复。

方法 1 的结果说明的是“小型 RA 闭集里的选择”，不是完整 MedDRA 自动编码。

## 方法 2：LLM 抽取 + 词法检索

该方法把语言理解与代码选择分开。

1. Qwen 只提取 adverse-event 原文 `quote/start` 和 assertion；抽取 JSON 中
   `code` 必须为空。exclusive `end` 同样由服务器产生并逐字验证。
2. 服务器在实际加载的术语索引中运行 exact LLT、local curated alias 和受限 fuzzy
   匹配。code 只能由检索器产生。
3. 每个候选返回 LLT、nullable PT、SOC、匹配片段、非概率 retrieval score 和
   provenance。
4. local alias 标记为 `LOCAL_DEMO_CURATED_ALIAS / UNVALIDATED`，默认需要人工确认；
   它不是 NCI 或 MedDRA 官方同义词。

## 方法 3：LLM + Hybrid RAG

该方法也不让 LLM 开放式生成 code。

1. Qwen 独立抽取事件和语境；offset 使用相同的 quote/start + 服务端逐字验证契约。
2. 每个事件在实际术语记录中计算标准 BM25、character-trigram overlap 和 exact
   boost，产生 Top-K 证据包。
3. 服务端记录候选集 hash 并冻结 Top-K。
4. Qwen 必须把所有候选 code 完整排列一次，且 `selected_code` 必须等于 rank 1。
5. 服务端拒绝任何新增、删除、重复或修改候选的响应；失败时不显示 code，并路由
   专家复核。

当前 “Hybrid” 指多种本地 lexical retrieval signals，不声称使用 dense embedding。
候选分数是检索相关度，不是校准概率，也不是临床准确率。

## 术语来源与 LLT/PT 层级

公开运行使用 NCI CTCAE v6.0 Clean Copy 中含 MedDRA 28.0 LLT Code、Term 和 SOC
字段的 850 行公开子集。服务器固定工作簿 SHA-256：

`4d7b4fcfdcb25c45a23b02c07fcb20eab7f284b23862e4e4e64bb8823c2f440b`

这不是公开或完整的 MedDRA 28.0 分发包。公共工作簿没有提供 Parent PT、HLT、
HLGT 和 SOC code，所以这些字段保持 `null`，绝不复制 LLT 或猜测父层级。

如果机构日后提供同版本、获授权的 `llt.asc`、`pt.asc`、`mdhier.asc`，启动时可通过
三个显式环境变量加载。只有解析出完整的 MedDRA 28.0 primary path 才会补入
PT/HLT/HLGT/SOC；26.1 与 28.0 不能混用。完整授权术语、派生索引和 credentials
不得进入 Git 或公开 HTML。

## 页面中应怎样读结果

每种方法卡片展示五步 trace：

1. 输入规则与 prompt hash；
2. 全部事件的逐字抽取和 assertion；
3. 每个事件自己的候选/冻结候选；
4. schema、allowlist、offset、候选集校验和审计 hash；
5. `AUTO_CANDIDATE`、`TOP_K_HUMAN_CHOICE` 或 `FULL_EXPERT_REVIEW`。

顶栏只有在至少一种方法实际返回 `LIVE` 时才显示 LIVE。全失败显示 ERROR；离线
静态页面显示 OFFLINE。PT 缺失时显示“公共来源未提供”，不会用 LLT 冒充。

## 数据、安全和评价边界

- 50 条记录及其预期 code 来自同一演示策展，只能报告 synthetic fixture
  agreement、schema 通过率、候选集违规率和延迟，不能报告临床 accuracy。
- 所有输出固定 `causality_not_assessed=true` 和
  `severity_not_assessed=true`；时间上发生在用药后也不代表药物导致事件。
- 否定、家族史、既往、已恢复、不确定、多事件或词表外记录必须进入人工路径。
- 演示输入仅限合成数据；不要粘贴真实患者资料或把 PHI 写入浏览器反馈备注。
- API 对 deidentified/restricted 请求不回传 top-level 原文或 raw model JSON；服务和
  vLLM 只监听 localhost，默认关闭 request access log，经 SSH tunnel 访问。
- 真实临床使用仍需要独立专家 reference、外部/前瞻验证、持续监测、数据治理、
  MedDRA 许可和本地安全审批。

## 可复现部署

Roihu 运行固定：

- model `Qwen/Qwen3.8-27B`
- revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- CSC `python-vllm/0.19.1`
- 单张 GH200，vLLM `127.0.0.1:8000`，API `127.0.0.1:8010`

详见 `docs/ROIHU_QWEN38_DEPLOYMENT.md`。模型 revision、prompt、候选集、术语来源、
术语 hash、run id、时间戳和 latency 均进入返回或审计字段。
