"""
Step 2: Global Schema Graph Construction
---------------------------------------

目标：统一不同来源的 schema（Spider tables.json 或真实 DB 的元数据）为表/字段异构图 G，
并注入金融领域特有的语义（代码字段的中文映射、隐式外键规则）。

核心输入：
- draw : 原始 schema 描述（Spider entry 或自定义字典列表）
- mmap : 字段代码 -> 中文描述映射表（可选）
- rlink: 隐式外键规则（字段名列表/集合，视为同名相连）

核心输出：
- tables: 归一化后的 Table 对象列表
- G     : SchemaGraph，包含表/字段节点与边（表-字段所属、字段-字段主/外键或隐式关联）
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -------------------------
# 数据结构
# -------------------------
@dataclass
class Column:
    name: str
    comment: str = ""
    dtype: str = ""
    table: "Table" | None = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    ref_table: Optional[str] = None
    ref_column: Optional[str] = None
    semantic: Optional[str] = None  # 通过 mmap 注入的中文语义


@dataclass
class Table:
    name: str
    comment: str = ""
    category: Optional[str] = None
    columns: List[Column] = field(default_factory=list)

    def add_column(self, column: Column) -> None:
        column.table = self
        self.columns.append(column)


@dataclass
class Edge:
    src: str
    dst: str
    etype: str  # "belongs_to" | "fk" | "implicit_fk"


class SchemaGraph:
    """简化的异构图：节点用字符串 ID 表示（table:tbl_name / column:tbl.col）。"""

    def __init__(self) -> None:
        self.nodes: set[str] = set()
        self.edges: List[Edge] = []
        self.adj: Dict[str, set[str]] = defaultdict(set)

    def add_node(self, node: str) -> None:
        self.nodes.add(node)

    def add_edge(self, src: str, dst: str, etype: str) -> None:
        self.add_node(src)
        self.add_node(dst)
        self.edges.append(Edge(src, dst, etype))
        self.adj[src].add(dst)
        self.adj[dst].add(src)

    def table_degree(self, table: Table) -> int:
        tid = f"table:{table.name}"
        return len(self.adj.get(tid, ()))

    def neighbors(self, node: str) -> Iterable[str]:
        return self.adj.get(node, ())


# -------------------------
# 适配器：Spider -> Table/Column
# -------------------------

def _load_from_spider_entry(entry: Dict, mmap: Optional[Dict[str, str]]) -> List[Table]:
    table_names: List[str] = entry.get("table_names_original") or entry.get("table_names") or []
    columns_raw: List[Tuple[int, str]] = entry.get("column_names_original") or entry.get("column_names") or []
    column_types: List[str] = entry.get("column_types") or []
    primary_keys = set(entry.get("primary_keys", ()))
    foreign_keys = entry.get("foreign_keys", [])

    tables = [Table(name=n) for n in table_names]
    cols: List[Column] = []
    for idx, (tbl_idx, col_name) in enumerate(columns_raw):
        if tbl_idx == -1:
            cols.append(None)  # type: ignore
            continue
        col = Column(name=col_name, dtype=column_types[idx] if idx < len(column_types) else "")
        if mmap and col.name in mmap:
            col.semantic = mmap[col.name]
        tables[tbl_idx].add_column(col)
        cols.append(col)

    # 标注主键
    for pk_idx in primary_keys:
        if 0 <= pk_idx < len(cols) and cols[pk_idx]:
            cols[pk_idx].is_primary_key = True

    # 标注外键
    for src_idx, dst_idx in foreign_keys:
        if 0 <= src_idx < len(cols) and 0 <= dst_idx < len(cols):
            src_col, dst_col = cols[src_idx], cols[dst_idx]
            if not src_col or not dst_col:
                continue
            src_col.is_foreign_key = True
            src_col.ref_table = dst_col.table.name if dst_col.table else None
            src_col.ref_column = dst_col.name
    return tables


# -------------------------
# 适配器：自定义 dict/list -> Table/Column
# -------------------------

def _load_from_dicts(draw: Sequence[Dict], mmap: Optional[Dict[str, str]]) -> List[Table]:
    tables: List[Table] = []
    for tbl in draw:
        t = Table(name=tbl.get("name"), comment=tbl.get("comment", ""), category=tbl.get("category"))
        for col in tbl.get("columns", []):
            c = Column(
                name=col.get("name"),
                comment=col.get("comment", ""),
                dtype=col.get("dtype", ""),
                is_primary_key=col.get("is_primary_key", False),
                is_foreign_key=col.get("is_foreign_key", False),
                ref_table=col.get("ref_table"),
                ref_column=col.get("ref_column"),
            )
            if mmap and c.name in mmap:
                c.semantic = mmap[c.name]
            t.add_column(c)
        tables.append(t)
    return tables


# -------------------------
# 核心：构建 SchemaGraph
# -------------------------

def build_schema_graph(
    draw,
    mmap: Optional[Dict[str, str]] = None,
    rlink: Optional[Sequence[str]] = None,
) -> tuple[List[Table], SchemaGraph]:
    """
    归一化 schema 并构建异构图 G。
    - draw : Spider entry dict 或包含表/字段信息的 dict 列表
    - mmap : 字段代码 -> 中文描述
    - rlink: 隐式外键字段名集合，如 ["Stkcd", "company_id"]
    """
    if isinstance(draw, dict) and (
        "table_names" in draw or "table_names_original" in draw
    ):
        tables = _load_from_spider_entry(draw, mmap)
    elif isinstance(draw, (list, tuple)):
        tables = _load_from_dicts(draw, mmap)
    else:
        raise TypeError("Unsupported schema format for build_schema_graph")

    graph = SchemaGraph()

    # 添加节点与表-字段边
    for table in tables:
        t_id = f"table:{table.name}"
        graph.add_node(t_id)
        for col in table.columns:
            c_id = f"column:{table.name}.{col.name}"
            graph.add_edge(t_id, c_id, "belongs_to")

    # 外键/主键边
    for table in tables:
        for col in table.columns:
            if col.is_foreign_key and col.ref_table and col.ref_column:
                src = f"column:{table.name}.{col.name}"
                dst = f"column:{col.ref_table}.{col.ref_column}"
                graph.add_edge(src, dst, "fk")

    # 隐式外键规则：同名字段互连
    if rlink:
        name_groups: Dict[str, List[Tuple[Table, Column]]] = defaultdict(list)
        ruleset = {n.lower() for n in rlink}
        for table in tables:
            for col in table.columns:
                if col.name.lower() in ruleset:
                    name_groups[col.name.lower()].append((table, col))
        for _, group in name_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    t1, c1 = group[i]
                    t2, c2 = group[j]
                    src = f"column:{t1.name}.{c1.name}"
                    dst = f"column:{t2.name}.{c2.name}"
                    graph.add_edge(src, dst, "implicit_fk")

    return tables, graph


# -------------------------
# 示例
# -------------------------
if __name__ == "__main__":  # pragma: no cover
    # Spider entry 示例
    spider_entry = {
        "table_names": ["balance_sheet", "company"],
        "column_names": [
            (-1, "*"),
            (0, "Stkcd"),
            (0, "A001101000"),
            (1, "company_id"),
            (1, "company_name"),
        ],
        "column_types": ["*", "text", "number", "text", "text"],
        "primary_keys": [1, 3],
        "foreign_keys": [(1, 3)],
    }
    mmap = {"A001101000": "货币资金"}
    tables, G = build_schema_graph(spider_entry, mmap=mmap, rlink=["Stkcd"])
    print("tables:", [t.name for t in tables])
    print("nodes:", sorted(G.nodes))
    print("edges:", [(e.src, e.dst, e.etype) for e in G.edges])
