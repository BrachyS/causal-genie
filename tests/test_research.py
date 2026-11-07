import json
from typing import cast, get_args

import pytest
from conftest import epmc_result, extraction
from pydantic import ValidationError
from tavily import TavilyClient

from causal_genie.agent.state import (
    CausalQuery,
    FirstRoundPlan,
    FollowUpPlan,
    Query,
    QueryGroup,
    ResearchRound,
    RunState,
)
from causal_genie.agent.steps.research import fallback_plan, missing_coverage, research
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.search import RESEARCH_DOMAINS, Searcher

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
def papers(make_epmc) -> EuropePMC:
    """Europe PMC that finds the D1 trial for any lookup."""
    return make_epmc([epmc_result()])[0]


# The fake Tavily returns one page for every query, so only round 1 has a new source to read.
READ = extraction(quotes=[])


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


def test_each_query_runs_with_its_tool(state, make_llm, searcher, papers) -> None:
    extract = q("primary_studies", "https://pubmed.ncbi.nlm.nih.gov/23921656/", tool="extract")
    llm, _ = make_llm([plan([*FULL_ROUND, extract], done=True), READ])

    research(state, llm, searcher, papers)

    assert len(searcher.calls) == len(GROUPS)
    assert all(call.domains == list(RESEARCH_DOMAINS) for call in searcher.calls)
    assert [e.url for e in searcher.extracts] == [extract["text"]]
    (only,) = state.rounds
    assert only.planned_by == "model" and len(only.plan.queries) == len(GROUPS) + 1
    (study,) = state.studies  # the snippet and the full page are one source, read in full
    assert study.full_text


def test_round_one_missing_a_group_falls_back_to_preset_plan(
    state, make_llm, searcher, papers
) -> None:
    missing = plan(FULL_ROUND[:-1], done=True)
    llm, fake = make_llm([missing, missing, READ])  # answer, repair attempt, then reading

    research(state, llm, searcher, papers)

    assert len(fake.requests) == 3  # one repair retry, no more; then the source is read
    (only,) = state.rounds
    assert only.planned_by == "fallback"
    assert {query.group for query in only.plan.queries} == set(GROUPS)
    assert llm.calls[0].error is not None  # the failed plan stays on record


def test_loop_continues_until_model_says_done(state, make_llm, searcher, papers) -> None:
    round_2 = [q("mechanism", "licorice glycyrrhizin anti-inflammatory")]
    llm, fake = make_llm([plan(FULL_ROUND), READ, plan(round_2, done=True)])

    research(state, llm, searcher, papers)

    assert len(state.rounds) == 2
    summary = fake.requests[2]["messages"][1]["content"]
    assert "Round 2 of 3" in summary
    assert FULL_ROUND[0]["text"] in summary  # queries already run
    assert "https://pubmed.ncbi.nlm.nih.gov/23921656/; snippet only" in summary  # studies


class OnePagePerQuery:
    """Fake Tavily that returns a different page for every query, so each search adds a study."""

    def search(self, query: str, **kwargs) -> dict:
        url = "https://example.org/" + query.replace(" ", "-")
        return {"results": [{"url": url, "title": query, "content": "...", "score": 0.5}]}


def distinct_studies(n: int, start: int) -> list[str]:
    """Extraction outputs for `n` unrelated included papers (no shared author or title)."""
    return [
        extraction(quotes=[], title=f"Unrelated study number {i}", first_author=f"Author{i}")
        for i in range(start, start + n)
    ]


def test_loop_stops_at_three_rounds(state, make_llm, make_epmc) -> None:
    searcher = Searcher(cast(TavilyClient, OnePagePerQuery()))
    round_2 = [q("confounders", f"c {i}") for i in range(2)]
    round_3 = [q("mechanism", f"m {i}") for i in range(2)]
    outcomes = [plan(FULL_ROUND), *distinct_studies(len(GROUPS), 0)]
    outcomes += [plan(round_2), *distinct_studies(2, 10), plan(round_3), *distinct_studies(2, 20)]
    llm, _ = make_llm(outcomes)

    research(state, llm, searcher, make_epmc([])[0])

    assert [r.new_studies for r in state.rounds] == [len(GROUPS), 2, 2]
    assert state.rounds[-1].stop_reason == "reached 3 rounds"


def test_round_adding_fewer_than_two_studies_stops(state, make_llm, searcher, papers) -> None:
    llm, _ = make_llm([plan(FULL_ROUND), READ, plan([q("mechanism", "licorice mechanism")])])

    research(state, llm, searcher, papers)

    assert [r.new_studies for r in state.rounds] == [1, 0]
    assert state.rounds[-1].stop_reason == "round 2 added 0 new studies (< 2)"


def test_done_with_coverage_met_stops(state, make_llm, searcher, papers) -> None:
    llm, _ = make_llm([plan(FULL_ROUND, done=True), READ])

    research(state, llm, searcher, papers)

    (only,) = state.rounds
    assert only.missing == [] and only.stop_reason == "model said done; coverage met"
    assert state.certainty is not None and state.certainty.level == "high"


def test_done_too_early_is_overridden_and_names_the_gap(state, make_llm, searcher, papers) -> None:
    news = extraction(quotes=[], source_kind="secondary")
    llm, fake = make_llm([plan(FULL_ROUND, done=True), news, plan([q("mechanism")], done=True)])

    research(state, llm, searcher, papers)

    assert len(state.rounds) == 2  # the model's first `done` was overridden
    (gap,) = state.rounds[0].missing
    assert "primary source" in gap
    summary = fake.requests[2]["messages"][1]["content"]
    assert "Coverage still missing" in summary and gap in summary


def test_nothing_found_reformulates_once_then_stops(state, make_llm, searcher, papers) -> None:
    off_topic = extraction(included=False, quotes=[])
    round_2 = [q("mechanism", "licorice root throat inflammation")]
    llm, fake = make_llm([plan(FULL_ROUND, done=True), off_topic, plan(round_2)])

    research(state, llm, searcher, papers)

    assert len(state.rounds) == 2
    assert "reformulate" in fake.requests[2]["messages"][1]["content"]
    assert state.rounds[-1].stop_reason == "round 2 added 0 new studies (< 2)"
    assert state.certainty is None


def test_repeated_queries_are_not_rerun_and_end_the_loop(state, make_llm, searcher, papers) -> None:
    llm, _ = make_llm([plan(FULL_ROUND), READ, plan([FULL_ROUND[0]])])

    research(state, llm, searcher, papers)

    assert len(searcher.calls) == len(GROUPS)  # the repeat was skipped
    assert len(state.rounds) == 2


def test_secondary_source_is_a_gap_only_without_an_equally_strong_paper(
    state, sample_state
) -> None:
    paper = sample_state.studies[0]  # an included RCT, read from the paper itself
    news_rct = paper.model_copy(update={"id": "n", "url": "https://news", "secondary_only": True})
    cohort = paper.model_copy(update={"id": "c", "url": "https://c", "design": "cohort"})
    state.rounds = [ResearchRound(number=1, plan=fallback_plan(state.query), planned_by="fallback")]

    state.studies = [news_rct, paper]
    assert missing_coverage(state) == []  # the RCT paper covers the news about an RCT

    state.studies = [news_rct, cohort]
    (gap,) = missing_coverage(state)
    assert "primary source" in gap and "https://news" in gap
