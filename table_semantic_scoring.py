"""Table-level semantic scoring (Step 2).

This module implements the semantic similarity matrix ``S_tbl`` between
question slots and database tables, as described in the step-by-step plan:

- Build a lightweight natural-language description for each table
  (name, comment, a few sample columns, category).
- Encode slots and table descriptions with an embedding model ``LM``.
- Compute cosine similarities to obtain ``S_tbl[i, j]``.

The code is framework-agnostic: provide any embedding model exposing an
``encode(text) -> vector`` method (e.g., ``text2vec``, ``bge``) and it will
produce the similarity matrix as a NumPy array.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Protocol, Sequence

import numpy as np


class EmbeddingModel(Protocol):
    """Minimal protocol for an embedding model used in Step 2."""

    def encode(self, text: str) -> Sequence[float]:  # pragma: no cover - protocol
        ...


@dataclass
class Column:
    """Schema column definition (simplified)."""

    name: str
    comment: str = ""
    dtype: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False


@dataclass
class Table:
    """Schema table definition (simplified)."""

    name: str
    comment: str = ""
    category: str = ""
    columns: List[Column] = field(default_factory=list)


def build_table_description(table: Table, max_columns: int = 5) -> str:
    """Construct a concise semantic description for a table.

    Args:
        table: Table metadata.
        max_columns: Max number of column names to include for context.

    Returns:
        A single string combining name, comment, sample columns, and category.
    """

    sample_cols = ", ".join(col.name for col in table.columns[:max_columns])
    comment = table.comment or "无"
    category = table.category or "unknown"
    parts = [
        f"表名:{table.name}",
        f"注释:{comment}",
        f"部分字段:{sample_cols}" if sample_cols else "部分字段:无",
        f"类型:{category}",
    ]
    return "; ".join(parts)


def _to_unit_vector(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return np.zeros_like(arr)
    return arr / norm


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Cosine similarity with zero-vector guard."""

    a = _to_unit_vector(vec_a)
    b = _to_unit_vector(vec_b)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def compute_table_semantic_scores(
    U: List[str],
    tables: List[Table],
    LM: EmbeddingModel,
    *,
    max_columns_in_desc: int = 5,
) -> np.ndarray:
    """Compute table-level semantic similarity matrix ``S_tbl``.

    Args:
        U: Slot list (typically one slot equal to the whole question in the
           current simplified pipeline).
        tables: All available tables.
        LM: Embedding model exposing ``encode(text) -> vector``.
        max_columns_in_desc: Number of column names to include when building
            table descriptions.

    Returns:
        ``S_tbl`` as an ``m x |T|`` NumPy array where
        ``S_tbl[i, j] = cosine_similarity(u_i, table_j)``.
    """

    m = len(U)
    p = len(tables)
    S_tbl = np.zeros((m, p), dtype=float)

    # Encode slots and tables once.
    slot_vecs = [_to_unit_vector(LM.encode(u)) for u in U]
    table_descs = [build_table_description(t, max_columns=max_columns_in_desc) for t in tables]
    table_vecs = [_to_unit_vector(LM.encode(desc)) for desc in table_descs]

    # Similarity matrix.
    for i, u_vec in enumerate(slot_vecs):
        for j, t_vec in enumerate(table_vecs):
            S_tbl[i, j] = cosine_similarity(u_vec, t_vec)

    return S_tbl


__all__ = [
    "Column",
    "Table",
    "EmbeddingModel",
    "build_table_description",
    "compute_table_semantic_scores",
    "cosine_similarity",
    "load_spider_schema",
]


def load_spider_schema(tables_json_path: str, db_id: str) -> List[Table]:
    """Load a Spider-format schema into ``Table`` objects.

    This helps you feed real database schemas into Step 2 without manual
    conversion. Spider's ``tables.json`` contains a list of database entries;
    we select the one matching ``db_id`` and convert its tables/columns.

    Args:
        tables_json_path: Path to ``tables.json`` from the Spider dataset.
        db_id: Database identifier inside Spider.

    Returns:
        A list of :class:`Table` objects with columns populated from Spider
        metadata. Column comments and table comments are not present in
        Spider, so the name is reused as the comment placeholder.
    """

    with open(tables_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    db_entry = next((e for e in entries if e.get("db_id") == db_id), None)
    if db_entry is None:
        raise ValueError(f"db_id={db_id!r} not found in {tables_json_path}")

    table_names = db_entry.get("table_names_original") or db_entry.get("table_names", [])
    column_names = db_entry.get("column_names_original") or db_entry.get("column_names", [])
    column_types = db_entry.get("column_types", [])
    primary_keys = set(db_entry.get("primary_keys", []))
    foreign_keys = db_entry.get("foreign_keys", [])

    # Build tables with empty column lists first.
    tables = [Table(name=tbl_name, comment=tbl_name, category="") for tbl_name in table_names]

    # Mark foreign-key columns for quick lookup.
    fk_columns = {fk_pair[0] for fk_pair in foreign_keys}

    for col_id, (tbl_idx, col_name) in enumerate(column_names):
        # Spider includes a synthetic column (-1, "*") representing all columns; skip it.
        if tbl_idx < 0 or tbl_idx >= len(tables):
            continue

        dtype = column_types[col_id] if col_id < len(column_types) else ""
        column = Column(
            name=col_name,
            comment=col_name,
            dtype=dtype,
            is_primary_key=col_id in primary_keys,
            is_foreign_key=col_id in fk_columns,
        )
        tables[tbl_idx].columns.append(column)

    return tables
