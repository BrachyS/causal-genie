import networkx as nx
import pytest

from causal_genie.agent.state import DagEdge
from causal_genie.tools.identify import (
    identify,
    precision_set,
    randomized_graph,
    valid_adjustment,
)


def edge(source: str, target: str, cited: bool = False) -> DagEdge:
    return DagEdge(source=source, target=target, claim_ids=["c1"] if cited else [])


def graph(*pairs: str) -> list[DagEdge]:
    """Edges from "a->b" strings."""
    return [edge(source, target) for source, target in (p.split("->") for p in pairs)]


def test_randomization_cuts_every_edge_into_the_exposure(step_logs) -> None:
    world = [
        edge("licorice_gargle", "sore_throat"),
        edge("smoking", "sore_throat"),
        edge("smoking", "licorice_gargle"),
        edge("age", "licorice_gargle"),
    ]

    trial = randomized_graph(world, exposure="licorice_gargle")

    assert trial == world[:2]
    assert "smoking -> licorice_gargle" in step_logs.text
    assert "age -> licorice_gargle" in step_logs.text


def test_graph_without_edges_into_the_exposure_is_unchanged() -> None:
    world = [edge("licorice_gargle", "sore_throat"), edge("smoking", "sore_throat")]

    assert randomized_graph(world, exposure="licorice_gargle") == world


# Textbook DAGs (PRD §5.7): x = exposure, y = outcome.
TEXTBOOK = [
    ("confounder", graph("c->x", "c->y", "x->y"), ["c"]),
    ("mediator: never adjust", graph("x->m", "m->y", "x->y"), []),
    ("collider: never adjust", graph("x->k", "y->k", "x->y"), []),
    ("M-bias: adjusting m opens a path", graph("a->x", "a->m", "b->m", "b->y", "x->y"), []),
    ("confounder plus mediator", graph("c->x", "c->y", "x->m", "m->y"), ["c"]),
]


@pytest.mark.parametrize(("case", "edges", "expected"), TEXTBOOK, ids=[t[0] for t in TEXTBOOK])
def test_textbook_dags(case, edges, expected) -> None:
    observed = sorted({n for e in edges for n in (e.source, e.target)})

    adj = identify(edges, "x", "y", observed, randomized=False)

    assert adj.adjust_for == expected
    assert adj.missing == []


def test_m_bias_set_stays_empty_when_only_the_collider_is_observed() -> None:
    edges = graph("a->x", "a->m", "b->m", "b->y", "x->y")

    assert identify(edges, "x", "y", ["x", "y", "m"], randomized=False).adjust_for == []


def test_unobserved_confounder_means_no_valid_set_and_names_what_is_missing() -> None:
    edges = graph("u->x", "u->y", "c->x", "c->y", "x->y")

    adj = identify(edges, "x", "y", ["x", "y", "c"], randomized=False)

    assert adj.adjust_for is None
    assert adj.missing == ["u"]


def test_randomized_data_needs_no_adjustment_even_with_unobserved_confounders() -> None:
    edges = graph("u->x", "u->y", "x->y")

    adj = identify(edges, "x", "y", ["x", "y"], randomized=True)

    assert (adj.adjust_for, adj.missing, adj.randomized) == ([], [], True)


def test_exposure_and_outcome_must_be_observed() -> None:
    with pytest.raises(ValueError, match="x"):
        identify(graph("x->y"), "x", "y", ["y"], randomized=False)


def digraph(edges: list[DagEdge]) -> nx.DiGraph:
    return nx.DiGraph([(e.source, e.target) for e in edges])


M_BIAS = graph("u1->x", "u1->z", "u2->z", "u2->y", "x->y")


@pytest.mark.parametrize(
    ("case", "edges", "adjust_for", "valid"),
    [
        ("confounder adjusted", graph("c->x", "c->y", "x->y"), ["c"], True),
        ("confounder left open", graph("c->x", "c->y", "x->y"), [], False),
        ("M-bias collider opens a path", M_BIAS, ["z"], False),
        ("M-bias collider with u1 closed", M_BIAS, ["z", "u1"], True),
        ("mediator is after the exposure", graph("x->m", "m->y", "x->y"), ["m"], False),
    ],
)
def test_valid_adjustment_is_the_backdoor_criterion(case, edges, adjust_for, valid) -> None:
    assert valid_adjustment(digraph(edges), "x", "y", adjust_for) is valid, case


# D0 with a correct DAG: nothing causes the moon, so the backdoor set is empty.
D0 = graph("full_moon->births", "weekday->births", "holiday->births", "season->births")


def test_outcome_causes_the_exposure_does_not_affect_are_added_for_precision() -> None:
    observed = ["full_moon", "births", "weekday", "holiday"]  # season not in the table

    assert precision_set(D0, "full_moon", "births", [], observed) == ["holiday", "weekday"]


def test_precision_skips_mediators_and_the_backdoor_set() -> None:
    edges = graph("c->x", "c->y", "x->m", "m->y", "x->y", "p->y")

    assert precision_set(edges, "x", "y", ["c"], ["x", "y", "c", "m", "p"]) == ["p"]


def test_precision_never_adds_a_variable_that_breaks_the_backdoor_criterion(
    monkeypatch: pytest.MonkeyPatch, step_logs
) -> None:
    from causal_genie.tools import identify as identify_module

    monkeypatch.setattr(identify_module, "valid_adjustment", lambda g, x, y, z: "holiday" not in z)

    observed = ["full_moon", "births", "weekday", "holiday"]
    assert precision_set(D0, "full_moon", "births", [], observed) == ["weekday"]
    assert "holiday" in step_logs.text
