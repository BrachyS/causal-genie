from causal_genie.agent.state import DagEdge
from causal_genie.tools.identify import randomized_graph


def edge(source: str, target: str) -> DagEdge:
    return DagEdge(source=source, target=target, claim_ids=[])


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
