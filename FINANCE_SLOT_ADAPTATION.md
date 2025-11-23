# 金融领域数据库适配与槽位抽取实践

本文档说明在不同金融数据库之间适配槽位抽取的推荐工程流程，包括别名词典的维护、抽取管线、LLM 兜底以及 schema 校验策略，并提供可用的数据集/模型参考。

## 1. 跨库/跨场景的别名与术语适配
- **维护多租户/多域词典**：按数据库或业务域分别维护 canonical schema（表/字段标准名）和别名映射（中文简称、英文、缩写、拼音/大小写变体、历史名称）。入口根据当前数据源加载对应词典。
- **自动扩充 + 人工校正**：从 schema 自动生成候选别名（驼峰拆分、下划线拆分、拼音），结合生产查询日志的高频短语，人工审核后入库。
- **消歧策略**：当同名出现在多个库/表时，结合请求的数据库上下文、表间连通性（能否形成可行 join）、字段类型约束，以及语义相似度得分，筛选 Top-1/Top-K 候选。
- **反馈闭环**：记录低置信度/被用户修正的别名，形成待审列表，周期性批量更新词典。

## 2. 推荐抽取管线（工程可落地）
1. **规则/词典粗召回**：使用业务词典 + 分词/匹配（含数字/日期正则）快速标注可能的实体、表、字段、时间、阈值等槽位，得到初始候选。
2. **模型精排/补全**：使用中文预训练模型微调的序列标注或 Span 模型，对输入问题进行槽位抽取；将规则候选作为先验特征或软约束，提高准确率。
3. **LLM 兜底**：当主模型置信度低或未覆盖到关键槽位时，调用 LLM 按固定 JSON 模板输出槽位；输出经过 schema 词典校验与纠错。
4. **schema 校验与筛选**：对每个槽位候选，要求能映射到当前词典中的 canonical 表/字段，类型必须合理（时间/数值/枚举），并优先选择能形成连通子图的组合；可保留 Top-K 及置信度供下游使用。
5. **上线与迭代**：灰度环境记录回退率和用户修正，扩充别名与标注样本，定期微调模型。

## 3. 数据集与模型参考（金融 NL2SQL）
- **数据集**：FinSQL（中文金融 NL2SQL），CSpider/DuSQL（中文通用 NL2SQL，可迁移），BIRD/BIRD-SQL（含商业智能场景），可用于初始微调与评测。
- **模型/检查点**：
  - 槽位抽取/语义匹配：`hfl/chinese-roberta-wwm-ext`、`ernie-3.0-base-zh`，可在金融语料或 FinSQL/CSpider 上微调。
  - LLM 兜底/SQL 生成：金融领域指令微调的 ChatGLM 版本、开源 CodeLlama/T5/BART 在 FinSQL/Spider 的权重。
  - Embedding：`text2vec-large-chinese`、`bge-large-zh`，可继续蒸馏金融数据。

## 4. 端到端工作流小结
- 入口确定目标数据库/业务域，加载对应 schema 词典和别名映射。
- 执行“规则/词典粗召回 → 模型抽取 → LLM 兜底 → schema 校验/消歧”的流水线，输出高置信槽位及 Top-K 候选。
- 使用表/字段连通性、类型约束与语义得分过滤不可用的组合；必要时仅返回高置信槽位，降低误导风险。
- 持续利用查询日志与用户反馈，更新别名词典和微调数据，提升跨库适配能力。

## 5. 快速落地示例：jieba/LTP + 通义点金 + 统一返回格式

**抽取方案**
- **粗召回**：使用 `jieba` 或 `LTP` 分词，加载数据库专属的别名词典（表/字段/实体常用词），通过匹配命中候选槽位。
- **LLM 兜底**：当粗召回缺失关键槽位或置信度不足时，调用通义点金让其按模板返回结构化结果，并在 prompt 中注入当前数据库的 canonical 名称/别名列表。
- **校验与映射**：所有槽位必须能映射到当前 schema 词典中的 canonical 表/字段；未通过校验的槽位降级为低置信或丢弃。

**推荐的返回数据结构（示例 JSON）**

```json
{
  "slots": [
    {
      "text": "净利润",               // 原始命中的片段
      "type": "metric",             // 槽位类型：table/column/entity/time/value/metric/operator 等
      "canonical": "finance.profit.net_income", // 规范化后的表/字段/实体标识
      "source": "dict|llm",         // 来源：粗召回(dict/jieba/ltp)或 LLM 兜底
      "confidence": 0.82             // 0~1 置信度
    },
    {
      "text": "2023Q4",
      "type": "time",
      "normalized": "2023-Q4",      // 可选：标准化后的时间或数值
      "source": "llm",
      "confidence": 0.77
    }
  ],
  "topk_candidates": {
    "table": [
      {"canonical": "finance.profit", "confidence": 0.86},
      {"canonical": "finance.balance_sheet", "confidence": 0.44}
    ],
    "column": [
      {"canonical": "finance.profit.net_income", "confidence": 0.82},
      {"canonical": "finance.profit.revenue", "confidence": 0.39}
    ]
  }
}
```

**使用要点**
- 统一 `canonical` 命名（库.表.字段或分层路径）便于下游对齐和 SQL 生成。
- 保留 `source` 与 `confidence`，方便后续重排、人工审核或兜底策略（如低置信槽位走二次 LLM 校验）。
- 可根据需要裁剪返回内容：只保留高置信槽位，或附带 Top-K 以便图连通性和类型约束进一步筛选。

## 6. 代码示例：可复用的混合槽位抽取模块
- `slot_extraction.py` 提供了可直接调用的工程代码：
  - `DictionarySlotExtractor`：基于 `jieba`/`LTP` + 词典的粗召回。
  - `ZhiZengSlotLLMClient`：使用智增增 OpenAI 兼容接口的兜底解析，prompt 已包含别名列表约束，默认模型 `qwen3-max`。
  - `SlotExtractionPipeline`：先跑词典召回，空结果时走 LLM，输出统一 JSON 结构。
- 使用方法参见 `README.md` 中的快速示例，配置智增增的 `base_url=https://api.zhizengzeng.com/v1/` 与 API Key 即可调用。
