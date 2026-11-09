"""Draw a DAG for people (PNG) and for programs (JSON).

Node names come from a model reading web pages, so they are only ever Graphviz labels: nodes
get generated ids and labels are escaped, so a name cannot add DOT syntax or HTML markup.
"""

import json
import logging
from pathlib import Path

import graphviz

from causal_genie.agent.state import DagEdge

log = logging.getLogger(__name__)


def dag_dot(edges: list[DagEdge]) -> graphviz.Digraph:
    """The Graphviz drawing of `edges`; an assumed edge is dashed, a cited one solid."""
    names = list(dict.fromkeys(name for e in edges for name in (e.source, e.target)))
    ids = {name: f"n{i}" for i, name in enumerate(names)}
    dot = graphviz.Digraph()
    for name in names:
        dot.node(ids[name], label=graphviz.escape(name))
    for e in edges:
        dot.edge(ids[e.source], ids[e.target], style="dashed" if e.assumed else None)
    return dot


def render_dag(edges: list[DagEdge], out_dir: Path) -> tuple[Path, Path]:
    """Write `dag.json` and `dag.png` to `out_dir`; needs Graphviz's `dot` program installed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dag.json"
    json_path.write_text(
        json.dumps({"edges": [e.model_dump() | {"assumed": e.assumed} for e in edges]}, indent=2)
    )
    png_path = out_dir / "dag.png"
    png_path.write_bytes(dag_dot(edges).pipe(format="png"))
    log.info("rendered dag: edges=%d json=%s png=%s", len(edges), json_path, png_path)
    return json_path, png_path
