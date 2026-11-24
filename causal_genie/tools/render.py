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

QUESTION_COLOR = "#8b5cf6"  # mid-tone purple: readable on light and dark backgrounds
# Filled nodes with dark text, so they read the same on the app's light and dark themes.
QUESTION_NODE = {"style": "filled", "fillcolor": "#ede9fe", "color": QUESTION_COLOR}
SOURCED_NODE = {"style": "filled", "fillcolor": "#d1fae5", "color": "#059669"}
ASSUMED_NODE = {"style": "filled,dashed", "fillcolor": "#f3f4f6", "color": "#6b7280"}
NODE_FONT = "#111827"


def node_style(name: str, edges: list[DagEdge]) -> dict[str, str]:
    """Where a node comes from: the question itself (exposure, outcome), the research (on at
    least one cited edge), or the model alone (only on assumed edges)."""
    if any(e.question and name in (e.source, e.target) for e in edges):
        return QUESTION_NODE
    if any(not e.assumed and name in (e.source, e.target) for e in edges):
        return SOURCED_NODE
    return ASSUMED_NODE


def dag_dot(edges: list[DagEdge]) -> graphviz.Digraph:
    """The Graphviz drawing of `edges`: assumed dashed, cited solid, and the question edge
    bold, purple and marked "?", since it is the hypothesis tested, not a cited cause. Nodes
    are colored by where they come from (`node_style`)."""
    names = list(dict.fromkeys(name for e in edges for name in (e.source, e.target)))
    ids = {name: f"n{i}" for i, name in enumerate(names)}
    dot = graphviz.Digraph()
    for name in names:
        dot.node(
            ids[name], label=graphviz.escape(name), fontcolor=NODE_FONT, **node_style(name, edges)
        )
    for e in edges:
        if e.question:
            dot.edge(
                ids[e.source],
                ids[e.target],
                style="bold",
                color=QUESTION_COLOR,
                fontcolor=QUESTION_COLOR,
                label=" ? ",
            )
        else:
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
