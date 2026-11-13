import json

import pytest

from causal_genie.agent.state import (
    ColumnProfile,
    DagEdge,
    DatasetCandidate,
    RunState,
    TableProfile,
)
from causal_genie.agent.steps.identify import map_columns, source


def column(name: str, dtype: str = "int64") -> ColumnProfile:
    return ColumnProfile(name=name, dtype=dtype, missing=0, unique=2, min=0, max=10, mean=1)


@pytest.fixture
def mapped_state(sample_state: RunState) -> RunState:
    """D1 after data discovery: DAG with smoking, the licorice table accepted and profiled."""
    state = sample_state.model_copy(deep=True)
    state.edges = [
        DagEdge(source="licorice_gargle", target="sore_throat", claim_ids=["c1"]),
        DagEdge(source="smoking", target="sore_throat", claim_ids=[]),
    ]
    state.datasets = [
        DatasetCandidate(
            url="https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/licorice_gargle.csv",
            host="vincentarelbundock.github.io",
            license="MIT",
            access="open",
            coverage={
                "licorice_gargle": "treat",
                "sore_throat": "pacu30min_throatPain",
                "smoking": "preOp_smoking",
            },  # fmt: skip
            score=1.0,
            rejected_reason=None,
            source="repository",
            title="licorice_gargle",
            description="235 rows × 19 columns",
            accepted=True,
        )
    ]
    state.profile = TableProfile(
        rows=235,
        columns=[
            column(c)
            for c in ("treat", "pacu30min_throatPain", "pacu90min_throatPain", "preOp_smoking")
        ],
    )
    return state


def node(name, columns, derive=None, cutoff=None, categorical=False) -> dict:
    return {"node": name, "columns": columns, "derive": derive, "cutoff": cutoff,
            "categorical": categorical}  # fmt: skip


def answer(*nodes: dict, unit=None, time=None) -> str:
    return json.dumps({"nodes": list(nodes), "unit": unit, "time": time})


GOOD = answer(
    node("licorice_gargle", ["treat"]),
    node("sore_throat", ["pacu30min_throatPain"], "cutoff", 0),
    node("smoking", ["preOp_smoking"], categorical=True),
)


def test_model_sees_column_summaries_and_the_coverage_guess_but_never_rows(
    make_llm, mapped_state
) -> None:
    llm, fake = make_llm([GOOD])

    map_columns(mapped_state, llm, ask=lambda q: "yes")

    user = fake.requests[0]["messages"][1]["content"]
    assert "pacu30min_throatPain" in user and "235 rows" in user
    assert "smoking: preOp_smoking" in user  # coverage from data discovery


def test_confirmed_mapping_is_kept_on_the_run(make_llm, mapped_state) -> None:
    llm, _ = make_llm([GOOD])
    asked = []

    mapping = map_columns(mapped_state, llm, ask=lambda q: asked.append(q) or "y")

    assert mapping is not None and mapped_state.mapping == mapping
    assert mapped_state.mapping_confirmed
    assert "sore_throat ← pacu30min_throatPain > 0" in asked[0]


def test_declined_mapping_is_dropped(make_llm, mapped_state) -> None:
    llm, _ = make_llm([GOOD])

    assert map_columns(mapped_state, llm, ask=lambda q: "no") is None
    assert mapped_state.mapping is None


def test_without_a_user_the_mapping_is_kept_unconfirmed(make_llm, mapped_state, step_logs) -> None:
    llm, _ = make_llm([GOOD])

    map_columns(mapped_state, llm, ask=None)

    assert mapped_state.mapping is not None and not mapped_state.mapping_confirmed
    assert "not confirmed" in step_logs.text


def test_harness_unmaps_bad_columns_and_rules_and_drops_unknown_nodes(
    make_llm, mapped_state, step_logs
) -> None:
    bad = answer(
        node("licorice_gargle", ["treat"]),
        node("sore_throat", ["pacu30min_throatPain"], "cutoff", None),  # cutoff missing
        node("smoking", ["smoker_flag"]),  # not a column
        node("age", ["preOp_age"]),  # not a DAG node
        unit="city",  # not a column
    )
    llm, _ = make_llm([bad])

    mapping = map_columns(mapped_state, llm, ask=None)

    assert mapping is not None
    assert [(m.node, m.columns) for m in mapping.nodes] == [
        ("licorice_gargle", ["treat"]),
        ("sore_throat", []),
        ("smoking", []),
    ]
    assert mapping.unit is None
    for logged in ("age", "smoker_flag", "cutoff", "city"):
        assert logged in step_logs.text


def test_node_the_model_skipped_is_unmapped(make_llm, mapped_state) -> None:
    llm, _ = make_llm([answer(node("licorice_gargle", ["treat"]))])

    mapping = map_columns(mapped_state, llm, ask=None)

    assert mapping is not None
    assert {m.node: m.columns for m in mapping.nodes} == {
        "licorice_gargle": ["treat"],
        "sore_throat": [],
        "smoking": [],
    }


def test_cutoff_may_span_several_columns(make_llm, mapped_state) -> None:
    columns = ["pacu30min_throatPain", "pacu90min_throatPain"]
    llm, _ = make_llm(
        [answer(node("licorice_gargle", ["treat"]), node("sore_throat", columns, "cutoff", 0))]
    )

    mapping = map_columns(mapped_state, llm, ask=None)

    assert mapping is not None
    sore = next(m for m in mapping.nodes if m.node == "sore_throat")
    assert sore.columns == columns
    assert source(sore) == "any of pacu30min_throatPain, pacu90min_throatPain > 0"
