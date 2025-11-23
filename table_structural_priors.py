"""Table-level structural priors (Step 3).

This module implements the structural prior matrix ``T_tbl`` between slots and
schema tables, as described in the step-by-step plan. It focuses on
lightweight, explainable heuristics:

- Query type detection (financial report / fund position / general).
- Table centrality in the schema graph.
- Presence of key entity identifiers in table columns (e.g., ``company_id``).
- Optional category match between inferred query type and table.category.

Provide a schema graph that exposes ``table_degree`` (and optionally
``table_neighbors`` for debugging/inspection). The included
:class:`SimpleSchemaGraph` builds a graph from a list of tables and undirected
edges between table names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol, Sequence, Tuple

import numpy as np

from table_semantic_scoring import Table


class SchemaGraph(Protocol):
    """Minimal protocol for a table-level schema graph used in Step 3."""

    def table_degree(self, table: Table) -> int:  # pragma: no cover - protocol
        ...


@dataclass
class SimpleSchemaGraph:
    """A lightweight table graph built from adjacency edges.

    Args:
        tables: List of ``Table`` objects (must have unique names).
        edges: Iterable of undirected connections between table names. Each
            tuple ``(a, b)`` indicates a relationship between ``a`` and ``b``.
    """

    tables: List[Table]
    edges: Iterable[Tuple[str, str]]

    def __post_init__(self) -> None:
        self._table_by_name: Dict[str, Table] = {t.name: t for t in self.tables}
        self._adj: Dict[str, set] = {t.name: set() for t in self.tables}
        for a, b in self.edges:
            if a not in self._adj or b not in self._adj:
                continue
            self._adj[a].add(b)
            self._adj[b].add(a)

    def table_degree(self, table: Table) -> int:
        return len(self._adj.get(table.name, ()))

    def table_neighbors(self, table: Table) -> Sequence[Table]:
        names = self._adj.get(table.name, ())
        return [self._table_by_name[n] for n in names if n in self._table_by_name]


def infer_query_type(q: str) -> str:
    """Heuristic query type detection for financial domains."""

    if any(k in q for k in ["净利润", "资产负债", "现金流", "ROE", "利润表"]):
        return "financial_report"
    if any(k in q for k in ["持仓", "重仓股", "基金"]):
        return "fund_position"
    return "general"


def compute_table_structural_priors(
    U: List[str],
    tables: List[Table],
    G: SchemaGraph,
    *,
    category_map: Dict[str, Sequence[str]] | None = None,
) -> np.ndarray:
    """Compute structural prior matrix ``T_tbl`` for slots vs tables.

    The scoring follows the reference pseudocode with small guardrails:
    - Table degree is transformed by ``log1p`` to reduce the effect of hubs.
    - Category matching can be customized via ``category_map``.
    - Key entity identifiers (``company_id`` / ``fund_id``) boost relevance.

    Args:
        U: Slot list (``m`` slots).
        tables: All tables in the schema.
        G: Schema graph exposing ``table_degree``.
        category_map: Optional mapping from query type to acceptable table
            categories. Defaults mirror the pseudocode.

    Returns:
        ``T_tbl`` as an ``m x |T|`` NumPy array.
    """

    if category_map is None:
        category_map = {
            "financial_report": {"profit", "balance_sheet", "cash_flow"},
            "fund_position": {"fund_basic", "fund_position"},
        }

    m = len(U)
    p = len(tables)
    T_tbl = np.zeros((m, p), dtype=float)

    # For simplicity we assume one slot represents the full question; reuse if >1.
    q_type = infer_query_type(U[0]) if U else "general"
    type_categories = category_map.get(q_type, set())

    for i, u in enumerate(U):
        for j, t in enumerate(tables):
            score = 0.0

            # (1) Table centrality (degree-based)
            degree = max(G.table_degree(t), 0)
            score += 0.1 * float(np.log1p(degree))

            # (2) Table category vs query type
            if t.category and (t.category in type_categories):
                score += 0.5

            # (3) Presence of key entities
            col_names = {col.name for col in t.columns}
            if "company_id" in col_names:
                score += 0.2
            if "fund_id" in col_names and ("基金" in u or q_type == "fund_position"):
                score += 0.3

            T_tbl[i, j] = score

    return T_tbl


__all__ = [
    "SchemaGraph",
    "SimpleSchemaGraph",
    "infer_query_type",
    "compute_table_structural_priors",
]
