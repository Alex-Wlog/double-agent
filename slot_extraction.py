"""
Slot extraction pipeline for financial databases.

Features:
- Dictionary-based recall using jieba or LTP tokenization.
- Canonical alias mapping for schema-aware normalization.
- Optional Zhizengzeng LLM fallback via OpenAI-compatible SDK.
- Unified output structure with per-slot confidence and source.

This module does not call remote services by default; provide a configured
OpenAI client to enable the LLM fallback.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class AliasEntry:
    """Alias metadata for a schema element."""

    alias: str
    canonical: str
    slot_type: str
    weight: float = 1.0


@dataclass
class Slot:
    """Structured slot output for downstream alignment."""

    text: str
    slot_type: str
    canonical: Optional[str]
    confidence: float
    source: str
    normalized: Optional[str] = None
    span: Optional[Tuple[int, int]] = None


class AliasDictionary:
    """Bidirectional mapping between aliases and canonical identifiers."""

    def __init__(self, entries: Iterable[AliasEntry]):
        self._alias_to_entry: Dict[str, AliasEntry] = {}
        self._canonical_to_aliases: Dict[str, List[str]] = {}
        for entry in entries:
            self._alias_to_entry[entry.alias] = entry
            self._canonical_to_aliases.setdefault(entry.canonical, []).append(entry.alias)

    def lookup(self, term: str) -> Optional[AliasEntry]:
        return self._alias_to_entry.get(term)

    def aliases_for_prompt(self, limit: int = 64) -> List[str]:
        aliases = list(self._alias_to_entry.keys())
        return aliases[:limit]


class DictionarySlotExtractor:
    """Dictionary-based recall using jieba or LTP tokenization."""

    def __init__(self, alias_dict: AliasDictionary, engine: str = "jieba"):
        self.alias_dict = alias_dict
        self.engine = engine
        self._tokenizer = self._load_tokenizer(engine)

    def _load_tokenizer(self, engine: str):
        if engine == "jieba":
            jieba = _require_module("jieba")
            for alias in self.alias_dict.aliases_for_prompt(limit=512):
                jieba.add_word(alias)
            return jieba
        if engine == "ltp":
            ltp_mod = _require_module("ltp")
            ltp_model = ltp_mod.LTP()  # type: ignore[attr-defined]
            for alias in self.alias_dict.aliases_for_prompt(limit=512):
                ltp_model.add_words([alias])
            return ltp_model
        raise ValueError(f"Unsupported tokenizer engine: {engine}")

    def tokenize(self, text: str) -> Sequence[str]:
        if self.engine == "jieba":
            return tuple(self._tokenizer.cut(text, HMM=False))
        seg, _ = self._tokenizer.seg([text])
        return tuple(seg[0])

    def extract(self, text: str, base_confidence: float = 0.7) -> List[Slot]:
        tokens = self.tokenize(text)
        slots: List[Slot] = []
        start_idx = 0
        for tok in tokens:
            alias_entry = self.alias_dict.lookup(tok)
            if alias_entry:
                end_idx = start_idx + len(tok)
                slots.append(
                    Slot(
                        text=tok,
                        slot_type=alias_entry.slot_type,
                        canonical=alias_entry.canonical,
                        confidence=min(1.0, base_confidence * alias_entry.weight),
                        source=self.engine,
                        span=(start_idx, end_idx),
                    )
                )
            start_idx += len(tok)
        return slots


class ZhiZengSlotLLMClient:
    """LLM fallback using Zhizengzeng (OpenAI-compatible) endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.zhizengzeng.com/v1/",
        model: str = "qwen3-max",
        base_confidence: float = 0.6,
    ):
        openai_mod = _require_module("openai")
        self.client = openai_mod.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_confidence = base_confidence

    def _build_prompt(self, question: str, alias_inventory: Sequence[str]) -> List[Dict[str, str]]:
        alias_lines = "\n".join(f"- {a}" for a in alias_inventory)
        user_prompt = f"""
你是金融数据库的槽位抽取助手。请从用户问题中提取涉及的表、字段、实体、时间或指标，返回 JSON。
仅使用下列已知别名/术语集合进行映射，不要编造：
{alias_lines}

输出 JSON 数组，每个元素包含: text, type, canonical(若无法映射则 null), confidence(0-1)。
问题: {question}
"""
        return [
            {"role": "system", "content": "You are a strict information extraction assistant."},
            {"role": "user", "content": user_prompt.strip()},
        ]

    def extract(self, question: str, alias_dict: AliasDictionary) -> List[Slot]:
        messages = self._build_prompt(question, alias_dict.aliases_for_prompt())
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content
        slots = []
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return slots
        for item in parsed if isinstance(parsed, list) else []:
            text = item.get("text")
            slot_type = item.get("type")
            canonical = item.get("canonical")
            confidence = float(item.get("confidence", self.base_confidence))
            alias_entry = alias_dict.lookup(canonical or text or "")
            normalized_canonical = canonical
            if alias_entry:
                normalized_canonical = alias_entry.canonical
            slots.append(
                Slot(
                    text=text or "",
                    slot_type=slot_type or "unknown",
                    canonical=normalized_canonical,
                    confidence=min(1.0, confidence),
                    source="llm",
                )
            )
        return slots


class SlotExtractionPipeline:
    """Hybrid pipeline: dictionary recall + optional LLM fallback + normalization."""

    def __init__(
        self,
        alias_entries: Iterable[AliasEntry],
        tokenizer_engine: str = "jieba",
        llm_client: Optional[ZhiZengSlotLLMClient] = None,
        confidence_threshold: float = 0.3,
    ):
        self.alias_dict = AliasDictionary(alias_entries)
        self.dictionary_extractor = DictionarySlotExtractor(self.alias_dict, engine=tokenizer_engine)
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

    def run(self, question: str) -> Dict[str, List[Dict[str, object]]]:
        slots = self.dictionary_extractor.extract(question)
        if self.llm_client and not slots:
            slots = self.llm_client.extract(question, self.alias_dict)
        filtered = [s for s in slots if s.confidence >= self.confidence_threshold]
        return {"slots": [slot.__dict__ for slot in filtered]}


def _require_module(module_name: str):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Module '{module_name}' is required but not installed."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = [
    "AliasEntry",
    "AliasDictionary",
    "DictionarySlotExtractor",
    "Slot",
    "SlotExtractionPipeline",
    "ZhiZengSlotLLMClient",
]
