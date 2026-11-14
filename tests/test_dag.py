import json
from typing import cast

import pytest
from conftest import proposal
from tavily import TavilyClient

from causal_genie.agent.state import DagEdge, ResearchPlan, ResearchRound, RunState
from causal_genie.agent.steps import dag
from causal_genie.agent.steps.dag import break_cycles, build_dag, run_dag
from causal_genie.llm.client import PROMPTS_DIR
from causal_genie.tools.derive import DATE_RULES
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.search import Searcher

QUESTION = ("licorice_gargle", "sore_throat")
NO_QUERIES = json.dumps({"queries": [], "done": True, "done_reason": "test"})


def edge(source: str, target: str, *claim_ids: str) -> DagEdge:
    """A DAG edge citing the given claims."""
    return DagEdge(source=source, target=target, claim_ids=list(claim_ids))


QUESTION_EDGE = DagEdge(source=QUESTION[0], target=QUESTION[1], claim_ids=[], question=True)


def test_edges_cite_ledger_claims_from_the_large_model(
    make_llm, sample_state: RunState, confounder_claim
) -> None:
    sample_state.claims.append(confounder_claim())
    llm, fake = make_llm(
        [
            proposal(
                ("licorice_gargle", "sore_throat", ["c1"]),
                ("smoking", "sore_throat", ["s1-adjusted_for"]),
            )
        ]
    )

    edges = build_dag(sample_state, llm)

    assert [(e.source, e.target, e.claim_ids) for e in edges] == [
        ("licorice_gargle", "sore_throat", []),  # the question edge cites nothing
        ("smoking", "sore_throat", ["s1-adjusted_for"]),
    ]
    assert fake.requests[0]["model"] == "test/large"
    sent = fake.requests[0]["messages"][1]["content"]
    assert "licorice_gargle" in sent and "sore_throat" in sent
    assert "s1-adjusted_for" in sent and "smokers were balanced" in sent


def test_edge_citing_an_unknown_claim_is_marked_assumed(
    make_llm, sample_state: RunState, step_logs
) -> None:
    llm, _ = make_llm(
        [
            proposal(
                ("age", "sore_throat", ["c1", "c99"]),
                ("smoking", "sore_throat", ["c42"]),
            )
        ]
    )

    edges = build_dag(sample_state, llm)

    assert edges[0].claim_ids == ["c1"]
    assert edges[1].claim_ids == [] and edges[1].assumed
    assert "c99" in step_logs.text and "c42" in step_logs.text


def test_unresolved_claims_are_not_shown_and_cannot_be_cited(
    make_llm, sample_state: RunState, confounder_claim
) -> None:
    sample_state.claims.append(confounder_claim(status="unresolved"))
    llm, fake = make_llm(
        [
            proposal(
                ("licorice_gargle", "sore_throat", ["c1"]),
                ("smoking", "sore_throat", ["s1-adjusted_for"]),
            )
        ]
    )

    edges = build_dag(sample_state, llm)

    assert "s1-adjusted_for" not in fake.requests[0]["messages"][1]["content"]
    assert edges[1].assumed


def test_missing_exposure_outcome_edge_is_added_as_the_question_edge(
    make_llm, sample_state: RunState
) -> None:
    llm, _ = make_llm([proposal(("age", "sore_throat", []))])

    edges = build_dag(sample_state, llm)

    assert [e for e in edges if (e.source, e.target) == QUESTION] == [QUESTION_EDGE]
    assert not QUESTION_EDGE.assumed


def test_claims_cited_on_the_question_edge_are_dropped_and_logged(
    make_llm, sample_state: RunState, step_logs
) -> None:
    llm, _ = make_llm([proposal((*QUESTION, ["c1"]))])

    edges = build_dag(sample_state, llm)

    assert edges == [QUESTION_EDGE]
    dropped = [r for r in step_logs.records if "hypothesis" in r.message]
    assert len(dropped) == 1 and "c1" in dropped[0].message


def test_model_cannot_mark_another_edge_as_the_question(make_llm, sample_state: RunState) -> None:
    edges = [
        {"source": "age", "target": "sore_throat", "claim_ids": [], "question": True},
        {"source": QUESTION[0], "target": QUESTION[1], "claim_ids": []},
    ]
    llm, _ = make_llm([json.dumps({"edges": edges})])

    built = build_dag(sample_state, llm)

    assert [(e.source, e.question, e.assumed) for e in built] == [
        ("age", False, True),
        ("licorice_gargle", True, False),
    ]


def test_near_copy_of_exposure_name_is_logged(make_llm, sample_state: RunState, step_logs) -> None:
    llm, _ = make_llm(
        [
            proposal(
                ("licorice_gargling", "sore_throat", ["c1"]),
                ("sore_throat_severity", "sore_throat", []),
            )
        ]
    )

    build_dag(sample_state, llm)

    renamed = [r.message for r in step_logs.records if "looks like" in r.message]
    assert len(renamed) == 1 and "licorice_gargling" in renamed[0]


def test_acyclic_graph_is_unchanged(step_logs) -> None:
    edges = [edge("age", "licorice_gargle"), edge("age", "sore_throat"), edge(*QUESTION, "c1")]

    assert break_cycles(edges, QUESTION) == edges
    assert "cycle" not in step_logs.text


def test_cycle_drops_its_assumed_edge_and_logs_it(step_logs) -> None:
    edges = [edge("a", "b", "c1"), edge("b", "c"), edge("c", "a", "c2"), edge(*QUESTION, "c3")]

    kept = break_cycles(edges, QUESTION)

    assert [(e.source, e.target) for e in kept] == [("a", "b"), ("c", "a"), QUESTION]
    dropped = [r.message for r in step_logs.records if "cycle" in r.message]
    assert len(dropped) == 1 and "b -> c" in dropped[0]


def test_cycle_drops_the_edge_with_fewest_citations() -> None:
    edges = [edge("a", "b", "c1", "c2"), edge("b", "a", "c3")]

    assert break_cycles(edges, QUESTION) == [edge("a", "b", "c1", "c2")]


def test_cycle_tie_is_broken_the_same_way_regardless_of_order() -> None:
    forward = [edge("a", "b"), edge("b", "a")]

    assert (
        break_cycles(forward, QUESTION) == break_cycles(forward[::-1], QUESTION) == [edge("b", "a")]
    )


def test_question_edge_survives_even_when_it_is_the_weakest() -> None:
    edges = [edge(*QUESTION), edge("sore_throat", "licorice_gargle", "c1")]

    assert break_cycles(edges, QUESTION) == [edge(*QUESTION)]


def test_every_cycle_is_broken_including_self_loops() -> None:
    edges = [edge("a", "a"), edge("b", "c"), edge("c", "b", "c1"), edge(*QUESTION, "c2")]

    assert break_cycles(edges, QUESTION) == [edge("c", "b", "c1"), edge(*QUESTION, "c2")]


def test_build_dag_returns_an_acyclic_graph(make_llm, sample_state: RunState) -> None:
    llm, _ = make_llm(
        [proposal((*QUESTION, ["c1"]), ("sore_throat", "age", []), ("age", "sore_throat", []))]
    )

    edges = build_dag(sample_state, llm)

    assert [(e.source, e.target) for e in edges] == [QUESTION, ("sore_throat", "age")]


@pytest.fixture
def researched(sample_state: RunState) -> RunState:
    """The sample run after one research round, before its DAG."""
    plan = ResearchPlan(queries=[], done=True, done_reason="test")
    sample_state.rounds.append(ResearchRound(number=1, plan=plan, planned_by="model"))
    sample_state.edges = []
    return sample_state


@pytest.fixture
def tools(fake_tavily, make_epmc) -> tuple[Searcher, EuropePMC]:
    """The search and paper-lookup clients `run_dag` passes on to research."""
    return Searcher(cast(TavilyClient, fake_tavily)), make_epmc([])[0]


def test_fully_cited_graph_needs_no_more_research(make_llm, researched: RunState, tools) -> None:
    llm, _ = make_llm([proposal((*QUESTION, ["c1"]))])

    run_dag(researched, llm, *tools)

    assert researched.edges == [QUESTION_EDGE]
    assert len(researched.rounds) == 1


def test_uncited_question_edge_alone_runs_no_extra_round(
    make_llm, researched: RunState, tools
) -> None:
    # One outcome only: an extra research round would ask the model for a plan and fail.
    llm, _ = make_llm([proposal((*QUESTION, []))])

    run_dag(researched, llm, *tools)

    assert researched.edges == [QUESTION_EDGE]
    assert len(researched.rounds) == 1


def test_uncited_edge_goes_back_to_research_and_the_graph_is_rebuilt(
    make_llm, researched: RunState, tools, confounder_claim, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm, _ = make_llm(
        [
            proposal(
                (*QUESTION, ["c1"]),
                ("smoking", "licorice_gargle", ["c1"]),
                ("smoking", "sore_throat", []),
            ),
            proposal(
                (*QUESTION, ["c1"]),
                ("smoking", "licorice_gargle", ["c1"]),
                ("smoking", "sore_throat", ["s1-adjusted_for"]),
            ),
        ]
    )
    asked: list[list[str]] = []

    def finds_a_claim(state: RunState, llm, searcher, papers, gaps: list[str], _) -> None:
        asked.append(gaps)
        state.claims.append(confounder_claim())

    monkeypatch.setattr(dag, "research_gaps", finds_a_claim)

    run_dag(researched, llm, *tools)

    assert asked == [["no source yet for DAG edge smoking -> sore_throat"]]
    assert not any(e.assumed for e in researched.edges)


def test_extra_research_finding_nothing_keeps_the_edge_assumed(
    make_llm, researched: RunState, tools
) -> None:
    # No third outcome: a second DAG call or another round would fail the test.
    llm, _ = make_llm(
        [
            proposal(
                (*QUESTION, ["c1"]), ("age", "licorice_gargle", []), ("age", "sore_throat", [])
            ),
            NO_QUERIES,
        ]
    )

    run_dag(researched, llm, *tools)

    assert len(researched.rounds) == 2
    assert [e.assumed for e in researched.edges] == [False, True, True]


def test_no_rounds_left_keeps_the_edge_assumed(make_llm, researched: RunState, tools) -> None:
    llm, _ = make_llm([proposal((*QUESTION, ["c1"]), ("age", "sore_throat", []))])

    run_dag(researched, llm, *tools, max_rounds=1)

    assert len(researched.rounds) == 1
    assert [e.assumed for e in researched.edges] == [False, True]


@pytest.fixture
def full_moon_state(sample_state: RunState) -> RunState:
    """D0: does a full moon cause more births? No claims, so every edge is assumed."""
    assert sample_state.query is not None
    query = sample_state.query.model_copy(
        update={
            "exposure": sample_state.query.exposure.model_copy(update={"name": "full_moon"}),
            "outcome": sample_state.query.outcome.model_copy(update={"name": "birth"}),
        }
    )
    return sample_state.model_copy(update={"query": query, "claims": []})


def test_edges_into_an_exogenous_node_are_dropped_and_edges_out_stay(
    make_llm, full_moon_state: RunState, step_logs
) -> None:
    # The D0 review (run 20261107-170133-b3826e): calendar factors drawn as causes of moon phase
    llm, _ = make_llm(
        [
            proposal(
                ("full_moon", "birth", []),
                ("season", "full_moon", []),
                ("holiday", "full_moon", []),
                ("weekend", "full_moon", []),
                ("weekend", "birth", []),
                ("full_moon", "daytime", []),
                exogenous=("full_moon",),
            )
        ]
    )

    edges = build_dag(full_moon_state, llm)

    assert [(e.source, e.target) for e in edges] == [
        ("full_moon", "birth"),
        ("weekend", "birth"),
        ("full_moon", "daytime"),
    ]
    for source in ("season", "holiday", "weekend"):
        assert f"exogenous full_moon: dropped {source} -> full_moon" in step_logs.text
    assert full_moon_state.exogenous == ["full_moon"]
    assert [(e.source, e.target) for e in full_moon_state.exogenous_dropped] == [
        ("season", "full_moon"),
        ("holiday", "full_moon"),
        ("weekend", "full_moon"),
    ]


def test_exogenous_name_not_in_the_graph_is_logged_and_ignored(
    make_llm, full_moon_state: RunState, step_logs
) -> None:
    llm, _ = make_llm(
        [proposal(("full_moon", "birth", []), ("weekend", "birth", []), exogenous=("weather",))]
    )

    edges = build_dag(full_moon_state, llm)

    assert [(e.source, e.target) for e in edges] == [("full_moon", "birth"), ("weekend", "birth")]
    assert "exogenous node weather is not in the graph; ignored" in step_logs.text
    assert full_moon_state.exogenous == [] and full_moon_state.exogenous_dropped == []


def test_question_edge_stays_even_if_the_outcome_is_called_exogenous(
    make_llm, full_moon_state: RunState, step_logs
) -> None:
    llm, _ = make_llm([proposal(("full_moon", "birth", []), exogenous=("birth",))])

    edges = build_dag(full_moon_state, llm)

    assert [(e.source, e.target, e.question) for e in edges] == [("full_moon", "birth", True)]
    assert "kept the question edge full_moon -> birth" in step_logs.text


def test_dag_prompt_asks_for_outcome_causes_the_data_may_record() -> None:
    """Precision variables come only from the DAG (`precision_set` reads it), so the prompt
    must ask for the outcome's main other causes, e.g. weekday for daily births (T10.9)."""
    text = (PROMPTS_DIR / "dag.md").read_text()

    assert "other causes of the outcome" in text
    assert "`day_of_week`" in text


def test_dag_prompt_names_calendar_nodes_like_the_derive_rules() -> None:
    """The data step covers a node named like a date rule from any table with a date (T10.9)."""
    text = (PROMPTS_DIR / "dag.md").read_text()

    for rule in DATE_RULES:
        assert f"`{rule}`" in text


def test_uncited_edges_from_outcome_only_nodes_run_no_extra_round(
    make_llm, researched: RunState, tools
) -> None:
    # Step 1 of T10.9 asks the DAG for the outcome's other causes (weekday, holiday for D0):
    # they only add precision, so finding a source for them is not worth a research round.
    # One outcome only: an extra round would ask the model for a plan and fail.
    llm, _ = make_llm(
        [
            proposal(
                (*QUESTION, ["c1"]), ("weekday", "sore_throat", []), ("holiday", "sore_throat", [])
            )
        ]
    )

    run_dag(researched, llm, *tools)

    assert len(researched.rounds) == 1
    assert [e.assumed for e in researched.edges] == [False, True, True]


def test_lone_uncited_edge_into_the_exposure_still_gets_a_research_round(
    make_llm, researched: RunState, tools
) -> None:
    # A possible instrument: IV detection needs this edge cited, so it is worth a round.
    llm, _ = make_llm(
        [proposal((*QUESTION, ["c1"]), ("distance", "licorice_gargle", [])), NO_QUERIES]
    )

    run_dag(researched, llm, *tools)

    assert len(researched.rounds) == 2


@pytest.fixture
def gaps_asked(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The gap lists `run_dag` sends to research; the round itself finds nothing."""
    asked: list[list[str]] = []

    def record(state: RunState, llm, searcher, papers, gaps: list[str], _) -> None:
        asked.append(gaps)

    monkeypatch.setattr(dag, "research_gaps", record)
    return asked


def test_d0_run4_uncited_edges_off_any_backdoor_path_run_no_round(
    make_llm, full_moon_state: RunState, tools, gaps_asked
) -> None:
    # Live D0 run 20261114-211437-12b84e spent ~50k tokens on a round for `season -> weather`.
    # Nothing causes the exposure, so no uncited edge can lie on a backdoor path.
    llm, _ = make_llm(
        [
            proposal(
                ("full_moon", "birth", []),
                ("day_of_week", "birth", []),
                ("season", "birth", []),
                ("year", "birth", []),
                ("weather", "birth", []),
                ("season", "weather", []),
                exogenous=("full_moon", "day_of_week", "season", "year"),
            )
        ]
    )

    run_dag(full_moon_state, llm, *tools)

    assert gaps_asked == []
    assert sum(e.assumed for e in full_moon_state.edges) == 5


def test_both_uncited_edges_of_a_confounder_are_sent(
    make_llm, researched: RunState, tools, gaps_asked
) -> None:
    llm, _ = make_llm(
        [proposal((*QUESTION, ["c1"]), ("age", "licorice_gargle", []), ("age", "sore_throat", []))]
    )

    run_dag(researched, llm, *tools)

    assert gaps_asked == [
        [
            "no source yet for DAG edge age -> licorice_gargle",
            "no source yet for DAG edge age -> sore_throat",
        ]
    ]


def test_uncited_edges_out_of_the_exposure_run_no_round(
    make_llm, researched: RunState, tools, gaps_asked
) -> None:
    # A mediator's edges never change the backdoor set (descendants are never adjusted for).
    llm, _ = make_llm(
        [
            proposal(
                (*QUESTION, ["c1"]),
                ("licorice_gargle", "throat_coating", []),
                ("throat_coating", "sore_throat", []),
            )
        ]
    )

    run_dag(researched, llm, *tools)

    assert gaps_asked == []
