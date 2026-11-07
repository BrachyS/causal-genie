import json
from typing import cast, get_args

import pytest
from pydantic import ValidationError
from tavily import TavilyClient

from causal_genie.agent.state import (
    CausalQuery,
    FirstRoundPlan,
    FollowUpPlan,
    Query,
    QueryGroup,
    RunState,
)
from causal_genie.agent.steps.research import fallback_plan, research
from causal_genie.tools.search import Searcher

GROUPS: tuple[str, ...] = get_args(QueryGroup)


def q(group: str, text: str | None = None, tool: str = "search") -> dict:
    """One planned query as the model would return it."""
    return {"text": text or f"licorice gargle {group}", "tool": tool, "group": group, "why": "w"}


def plan(queries: list[dict], done: bool = False) -> str:
    """A research plan as model output JSON."""
    return json.dumps({"queries": queries, "done": done, "done_reason": "test"})


FULL_ROUND = [q(g) for g in GROUPS]


@pytest.fixture
def state(sample_state: RunState) -> RunState:
    """The sample run, framed, before any research."""
    return RunState(run_id="r1", question=sample_state.question, query=sample_state.query)


@pytest.fixture
def searcher(fake_tavily) -> Searcher:
    """A live searcher over the fake Tavily."""
    return Searcher(cast(TavilyClient, fake_tavily))


def test_first_round_plan_must_cover_every_group() -> None:
    queries = [q(g) for g in GROUPS if g != "evidence_against"]

    with pytest.raises(ValidationError, match="evidence_against"):
        FirstRoundPlan.model_validate_json(plan(queries))


def test_first_round_plan_has_at_most_12_queries() -> None:
    queries = FULL_ROUND + [q("primary_studies", f"extra {i}") for i in range(7)]

    with pytest.raises(ValidationError, match="12"):
        FirstRoundPlan.model_validate_json(plan(queries))


def test_follow_up_plan_has_at_most_6_queries() -> None:
    with pytest.raises(ValidationError, match="6"):
        FollowUpPlan.model_validate_json(plan([q("mechanism", f"m {i}") for i in range(7)]))


def test_extract_query_must_be_a_url() -> None:
    with pytest.raises(ValidationError, match="URL"):
        Query.model_validate(q("primary_studies", "licorice trial", tool="extract"))


def test_fallback_plan_covers_every_group_with_short_terms(state: RunState) -> None:
    p = fallback_plan(cast(CausalQuery, state.query))

    assert {query.group for query in p.queries} == set(GROUPS)
    assert all("licorice gargle" in query.text for query in p.queries)
    FirstRoundPlan.model_validate(p.model_dump())  # passes the controller's own check


def test_each_query_runs_with_its_tool(state, make_llm, searcher) -> None:
    extract = q("primary_studies", "https://pubmed.ncbi.nlm.nih.gov/23921656/", tool="extract")
    llm, _ = make_llm([plan([*FULL_ROUND, extract], done=True)])

    research(state, llm, searcher)

    assert len(searcher.calls) == len(GROUPS)
    assert [e.url for e in searcher.extracts] == [extract["text"]]
    (only,) = state.rounds
    assert only.planned_by == "model" and len(only.plan.queries) == len(GROUPS) + 1


def test_round_one_missing_a_group_falls_back_to_preset_plan(state, make_llm, searcher) -> None:
    missing = plan(FULL_ROUND[:-1], done=True)
    llm, fake = make_llm([missing, missing])  # first answer, then the repair attempt

    research(state, llm, searcher)

    assert len(fake.requests) == 2  # one repair retry, no more
    (only,) = state.rounds
    assert only.planned_by == "fallback"
    assert {query.group for query in only.plan.queries} == set(GROUPS)
    assert llm.calls[0].error is not None  # the failed plan stays on record


def test_loop_continues_until_model_says_done(state, make_llm, searcher) -> None:
    round_2 = [q("mechanism", "licorice glycyrrhizin anti-inflammatory")]
    llm, fake = make_llm([plan(FULL_ROUND), plan(round_2, done=True)])

    research(state, llm, searcher)

    assert len(state.rounds) == 2
    summary = fake.requests[1]["messages"][1]["content"]
    assert "Round 2 of 3" in summary
    assert FULL_ROUND[0]["text"] in summary  # queries already run
    assert "https://pubmed.ncbi.nlm.nih.gov/23921656/" in summary  # sources found


def test_loop_stops_at_three_rounds(state, make_llm, searcher) -> None:
    rounds = [plan(FULL_ROUND)] + [plan([q("confounders", f"c {i}")]) for i in range(2)]
    llm, fake = make_llm(rounds)

    research(state, llm, searcher)

    assert len(state.rounds) == 3
    assert len(fake.requests) == 3


def test_repeated_queries_are_not_rerun_and_end_the_loop(state, make_llm, searcher) -> None:
    llm, _ = make_llm([plan(FULL_ROUND), plan([FULL_ROUND[0]])])

    research(state, llm, searcher)

    assert len(searcher.calls) == len(GROUPS)  # the repeat was skipped
    assert len(state.rounds) == 2
