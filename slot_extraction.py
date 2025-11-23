"""
Phase 1: Query Parsing & Initialization
--------------------------------------
将非结构化的用户问题转化为结构化的语义单元，输出槽位集合 U 和关键词集合 Kq。

核心策略：规则 + 词典 + LLM 兜底
- 规则：正则抓取时间表达、数字、简单操作符。
- 词典：指标编码映射、实体同义词映射、业务关键词词表，用于粗召回和标准化。
- LLM 兜底：通过 OpenAI 兼容接口（智增增 base_url + `qwen3-max`）补全遗漏槽位。

依赖：
- 可选 `jieba` 分词（未安装时自动降级为简单按空格/字符切分）。
- `openai` 官方 SDK（新版 `OpenAI` 客户端），用于 LLM 兜底。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import jieba
except Exception:  # pragma: no cover - 仅用于可选依赖降级
    jieba = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - 仅用于无 LLM 依赖时
    OpenAI = None


# -------------------------
# 数据结构
# -------------------------
@dataclass
class Slot:
    """语义槽位，包含类型、值、来源、置信度及可选原始片段。"""

    type: str
    value: str
    source: str
    confidence: float
    raw: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": self.type,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "raw": self.raw,
        }


@dataclass
class KnowledgeBase:
    """领域知识库/词典集合。"""

    metric_map: Dict[str, str] = field(default_factory=dict)  # code -> display name
    metric_alias: Dict[str, str] = field(default_factory=dict)  # alias -> code
    entity_alias: Dict[str, str] = field(default_factory=dict)  # alias -> entity_id
    keyword_whitelist: Sequence[str] = field(default_factory=tuple)
    time_patterns: Sequence[str] = field(
        default_factory=lambda: (
            r"\d{4}年?(?:Q[1-4])?",
            r"20\d{2}[-/](?:0?[1-9]|1[0-2])",
            r"20\d{2}",
        )
    )

    def find_metric(self, token: str) -> Optional[Slot]:
        code = self.metric_alias.get(token)
        if code:
            name = self.metric_map.get(code, token)
            return Slot("metric", code, "dict", 0.95, raw=name)
        if token in self.metric_map:
            return Slot("metric", token, "dict", 0.9, raw=self.metric_map[token])
        return None

    def find_entity(self, token: str) -> Optional[Slot]:
        ent = self.entity_alias.get(token)
        if ent:
            return Slot("entity", ent, "dict", 0.9, raw=token)
        return None

    def iter_time_matches(self, text: str) -> Iterable[Slot]:
        for pat in self.time_patterns:
            for m in re.finditer(pat, text):
                yield Slot("time", m.group(0), "regex", 0.9)


# -------------------------
# 核心流程
# -------------------------

def tokenize(text: str) -> List[str]:
    """使用 jieba（如可用）或回退到空格/逐字切分。"""
    if jieba:
        return [t for t in jieba.cut(text) if t.strip()]
    if " " in text:
        return [t for t in text.split(" ") if t]
    return list(text)


def extract_with_rules_and_dicts(q: str, kb: KnowledgeBase) -> List[Slot]:
    slots: List[Slot] = []

    # 时间正则
    for slot in kb.iter_time_matches(q):
        slots.append(slot)

    # 分词 + 词典召回
    for token in tokenize(q):
        metric = kb.find_metric(token)
        if metric:
            slots.append(metric)
            continue
        entity = kb.find_entity(token)
        if entity:
            slots.append(entity)

    # 去重（type, value）维度
    uniq = {}
    for s in slots:
        key = (s.type, s.value)
        if key not in uniq or uniq[key].confidence < s.confidence:
            uniq[key] = s
    return list(uniq.values())


def build_keyword_set(q: str, slots: Sequence[Slot], kb: KnowledgeBase) -> List[str]:
    kw = set(kb.keyword_whitelist)
    for s in slots:
        if s.raw:
            kw.add(s.raw)
        kw.add(s.value)
    # 加入原问题中的名词片段/高频分词
    kw.update(tokenize(q))
    return list(kw)


# -------------------------
# LLM 兜底（智增增 OpenAI 兼容接口）
# -------------------------

def make_zhizeng_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    if OpenAI is None:
        raise ImportError("openai package is required for LLM fallback")
    api_key = api_key or os.getenv("ZHIZENG_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("ZHIZENG_BASE_URL") or "https://api.zhizengzeng.com/v1"
    return OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))


def call_llm_fallback(q: str, kb: KnowledgeBase, client=None) -> List[Slot]:
    client = client or make_zhizeng_client()
    system_prompt = (
        "你是金融领域的槽位抽取助手。给定用户问题，提取时间(time)、实体(entity)、指标(metric)三类槽位。"
        "输出 JSON 数组，每个元素包含 type (time/entity/metric) 和 value。尽量用标准化编码："
        "指标用 metric_map 或 metric_alias 中的编码，实体用 entity_alias 映射后的 ID，找不到则返回原文。"
    )
    kb_hint = {
        "metric_alias": list(kb.metric_alias.items())[:50],
        "entity_alias": list(kb.entity_alias.items())[:50],
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps({"query": q, "dict": kb_hint}, ensure_ascii=False),
        },
    ]
    resp = client.chat.completions.create(
        model="qwen3-max",
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    slots: List[Slot] = []
    for item in payload.get("slots", []):
        if not isinstance(item, dict):
            continue
        slot_type = item.get("type")
        value = item.get("value")
        if slot_type in {"time", "entity", "metric"} and isinstance(value, str):
            slots.append(Slot(slot_type, value, "llm", 0.55, raw=item.get("raw")))
    return slots


# -------------------------
# 对外主函数
# -------------------------

def extract_semantic_units(
    q: str,
    kb: KnowledgeBase,
    use_llm_fallback: bool = True,
    client=None,
) -> Dict[str, object]:
    """
    主入口：返回 {"slots": U, "keywords": Kq}
    - 先用规则+词典召回
    - 槽位不足时可选调用 LLM 兜底
    - 最终去重并生成关键词集合
    """
    rule_slots = extract_with_rules_and_dicts(q, kb)
    slots = list(rule_slots)

    need_llm = use_llm_fallback and len(slots) < 2
    if need_llm:
        llm_slots = call_llm_fallback(q, kb, client=client)
        # 合并 LLM 槽位，若规则已有则只保留置信度更高的
        merged = {(s.type, s.value): s for s in slots}
        for s in llm_slots:
            key = (s.type, s.value)
            if key not in merged or merged[key].confidence < s.confidence:
                merged[key] = s
        slots = list(merged.values())

    keywords = build_keyword_set(q, slots, kb)
    return {
        "slots": [s.to_dict() for s in slots],
        "keywords": keywords,
    }


if __name__ == "__main__":  # 简单示例
    kb = KnowledgeBase(
        metric_map={"A001101000": "货币资金"},
        metric_alias={"货币资金": "A001101000", "现金": "A001101000"},
        entity_alias={"茅台": "600519", "贵州茅台": "600519"},
        keyword_whitelist=("同比", "环比"),
    )
    question = "2023年茅台的净利润和货币资金是多少"
    result = extract_semantic_units(question, kb, use_llm_fallback=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
