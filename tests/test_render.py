import json
import shutil
from pathlib import Path

import graphviz
import pytest

from causal_genie.agent.state import DagEdge
from causal_genie.tools.render import dag_dot, render_dag

CITED = DagEdge(source="age", target="sore_throat", claim_ids=["c1"])
ASSUMED = DagEdge(source="licorice_gargle", target="sore_throat", claim_ids=[])
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def edge_lines(source: str) -> list[str]:
    """The lines of DOT source that draw an edge."""
    return [line for line in source.splitlines() if "->" in line]


def test_assumed_edges_are_dashed_and_cited_edges_solid() -> None:
    lines = edge_lines(dag_dot([CITED, ASSUMED]).source)

    assert len(lines) == 2
    assert "dashed" not in lines[0]
    assert "style=dashed" in lines[1]


def test_each_node_is_drawn_once_with_its_name() -> None:
    source = dag_dot([CITED, ASSUMED]).source

    assert source.count("label=sore_throat") == 1
    assert source.count("label=age") == 1


def test_node_names_cannot_inject_dot_syntax() -> None:
    hostile = DagEdge(source='a:b:n[image="/etc/passwd"]', target="<b>x</b>", claim_ids=[])

    assert dag_dot([hostile]).source.splitlines()[1:-1] == [
        "\t" + r'n0 [label="a:b:n[image=\"/etc/passwd\"]"]',
        '\tn1 [label="<b>x</b>"]',
        "\tn0 -> n1 [style=dashed]",
    ]


def test_render_writes_json_and_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graphviz.Digraph, "pipe", lambda self, format: PNG_SIGNATURE)

    json_path, png_path = render_dag([CITED, ASSUMED], tmp_path / "run")

    assert json.loads(json_path.read_text()) == {
        "edges": [
            {"source": "age", "target": "sore_throat", "claim_ids": ["c1"], "assumed": False},
            {
                "source": "licorice_gargle",
                "target": "sore_throat",
                "claim_ids": [],
                "assumed": True,
            },
        ]
    }
    assert png_path.read_bytes() == PNG_SIGNATURE


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz `dot` is not installed")
def test_real_graphviz_writes_a_png(tmp_path: Path) -> None:
    _, png_path = render_dag([CITED, ASSUMED], tmp_path)

    assert png_path.read_bytes().startswith(PNG_SIGNATURE)
