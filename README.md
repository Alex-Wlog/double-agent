# double-agent

Slot extraction sandbox for financial databases. This repo now includes a hybrid
pipeline that combines dictionary recall (jieba/LTP) with an optional Tongyi
(Qwen) fallback and unified JSON output.

We also provide a Step 2 implementation for table-level semantic scoring
(`table_semantic_scoring.py`).

## Usage

```python
from slot_extraction import (
    AliasEntry,
    SlotExtractionPipeline,
    TongyiSlotLLMClient,
)
from openai import OpenAI

# 1) Build alias dictionary (table/field/entity aliases -> canonical names)
alias_entries = [
    AliasEntry(alias="净利润", canonical="finance.profit.net_income", slot_type="metric"),
    AliasEntry(alias="基金代码", canonical="fund.basic.code", slot_type="column"),
]

# 2) (Optional) Configure Tongyi LLM client for fallback
# client = OpenAI(api_key="<DASHSCOPE_API_KEY>", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# llm_client = TongyiSlotLLMClient(client)
llm_client = None

# 3) Run the pipeline
pipeline = SlotExtractionPipeline(alias_entries, tokenizer_engine="jieba", llm_client=llm_client)
result = pipeline.run("2023年基金代码是多少？")
print(result)
```

## Step 2: Table-level semantic scoring (S_tbl)

```python
import numpy as np
from table_semantic_scoring import (
    Column,
    Table,
    compute_table_semantic_scores,
    load_spider_schema,
)


class DummyEmbedder:
    def encode(self, text: str):
        # Replace with a real embedding model (e.g., text2vec, bge)
        return np.array([len(text), 1.0])


tables = [
    Table(
        name="finance_profit",
        comment="利润表",
        category="profit",
        columns=[Column(name="net_income"), Column(name="revenue")],
    ),
    Table(
        name="fund_position",
        comment="基金持仓",
        category="fund_position",
        columns=[Column(name="fund_code"), Column(name="holding_ratio")],
    ),
]

slots = ["2023年净利润"]
S_tbl = compute_table_semantic_scores(slots, tables, LM=DummyEmbedder())
print(S_tbl)
```

### Loading a real schema from Spider tables.json

```python
from table_semantic_scoring import load_spider_schema, compute_table_semantic_scores

tables = load_spider_schema("/path/to/tables.json", db_id="database_id")
slots = ["2020年净利润"]
S_tbl = compute_table_semantic_scores(slots, tables, LM=DummyEmbedder())
```

## Notes
- Install a tokenizer: `pip install jieba` or `pip install ltp` (choose one and
  set `tokenizer_engine` accordingly).
- To enable LLM fallback, install the OpenAI-compatible SDK (`pip install
  openai`) and provide `DASHSCOPE_API_KEY` plus the DashScope base URL shown
  above.
- Output is always `{"slots": [...]}` with per-slot confidence, canonical name,
  and source (`jieba`/`ltp`/`llm`).

## Financial text2sql datasets to start with

- **FinSQL**: Chinese financial QA-to-SQL pairs（财报、行情、指标等）。
- **CSpider / DuSQL**: 中文 NL2SQL 基准，非纯金融但可迁移结构知识。
- **BIRD/BIRD-SQL**: 商业智能方向（含金融子域，英文为主）。
- **FIBEN / FIBEN-SQL**（若可获取）：企业/金融商业分析问答，部分社区再发布版本。
- 上述可用于冷启动，结合自家 schema 做少量标注微调。
