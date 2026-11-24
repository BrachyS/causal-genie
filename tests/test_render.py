import json
import shutil
from pathlib import Path

import graphviz
import pytest

from causal_genie.agent.state import DagEdge
from causal_genie.tools.render import (
    ASSUMED_NODE,
    QUESTION_COLOR,
    QUESTION_NODE,
    SOURCED_NODE,
    dag_dot,
    render_dag,
)

CITED = DagEdge(source="age", target="sore_throat", claim_ids=["c1"])
ASSUMED = DagEdge(source="licorice_gargle", target="sore_throat", claim_ids=[])
QUESTION = DagEdge(source="smoking", target="sore_throat", claim_ids=[], question=True)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def node_line(source: str, name: str) -> str:
    """The DOT line that draws the node labelled `name`."""
    (line,) = [line for line in source.splitlines() if f"label={name}" in line]
    return line


def edge_lines(source: str) -> list[str]:
    """The lines of DOT source that draw an edge."""
    return [line for line in source.splitlines() if "->" in line]


def test_assumed_edges_are_dashed_and_cited_edges_solid() -> None:
    lines = edge_lines(dag_dot([CITED, ASSUMED]).source)

    assert len(lines) == 2
    assert "dashed" not in lines[0]
    assert "style=dashed" in lines[1]


def test_question_edge_is_drawn_as_a_question_not_a_finding() -> None:
    (line,) = edge_lines(dag_dot([QUESTION]).source)

    # bold, its own color and a "?": the hypothesis tested, not a cited cause
    assert "style=bold" in line and "dashed" not in line
    assert f'color="{QUESTION_COLOR}"' in line and 'label=" ? "' in line


def test_other_edges_get_no_question_mark() -> None:
    assert all("label" not in line for line in edge_lines(dag_dot([CITED, ASSUMED]).source))


def test_each_node_is_drawn_once_with_its_name() -> None:
    source = dag_dot([CITED, ASSUMED]).source

    assert source.count("label=sore_throat") == 1
    assert source.count("label=age") == 1


def test_node_names_cannot_inject_dot_syntax() -> None:
    hostile = DagEdge(source='a:b:n[image="/etc/passwd"]', target="<b>x</b>", claim_ids=[])

    lines = dag_dot([hostile]).source.splitlines()[1:-1]
    assert lines[0].startswith("\t" + r'n0 [label="a:b:n[image=\"/etc/passwd\"]"')
    assert lines[1].startswith('\tn1 [label="<b>x</b>"')
    assert lines[2] == "\tn0 -> n1 [style=dashed]"


def test_nodes_are_colored_by_where_they_come_from() -> None:
    """Question nodes (exposure, outcome) stand out; a node on a cited edge came from the
    research; a node only on assumed edges is the model's own idea."""
    source = dag_dot([QUESTION, CITED, DagEdge(source="mood", target="age", claim_ids=[])]).source

    assert f'fillcolor="{QUESTION_NODE["fillcolor"]}"' in node_line(source, "smoking")
    assert f'fillcolor="{QUESTION_NODE["fillcolor"]}"' in node_line(source, "sore_throat")
    assert f'fillcolor="{SOURCED_NODE["fillcolor"]}"' in node_line(source, "age")
    mood = node_line(source, "mood")
    assert f'fillcolor="{ASSUMED_NODE["fillcolor"]}"' in mood and "dashed" in mood


def test_a_question_node_keeps_its_color_when_it_also_has_cited_edges() -> None:
    source = dag_dot([QUESTION, DagEdge(source="smoking", target="age", claim_ids=["c1"])]).source

    assert f'fillcolor="{QUESTION_NODE["fillcolor"]}"' in node_line(source, "smoking")


def test_render_writes_json_and_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graphviz.Digraph, "pipe", lambda self, format: PNG_SIGNATURE)

    json_path, png_path = render_dag([CITED, ASSUMED], tmp_path / "run")

    assert json.loads(json_path.read_text()) == {
        "edges": [
            {
                "source": "age",
                "target": "sore_throat",
                "claim_ids": ["c1"],
                "question": False,
                "assumed": False,
            },
            {
                "source": "licorice_gargle",
                "target": "sore_throat",
                "claim_ids": [],
                "question": False,
                "assumed": True,
            },
        ]
    }
    assert png_path.read_bytes() == PNG_SIGNATURE


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz `dot` is not installed")
def test_real_graphviz_writes_a_png(tmp_path: Path) -> None:
    _, png_path = render_dag([CITED, ASSUMED], tmp_path)

    assert png_path.read_bytes().startswith(PNG_SIGNATURE)
