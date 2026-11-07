"""Research loop (PRD §5.3): the model plans each round; the harness runs it and decides to stop.

The model writes the queries, picks `search` or `extract` for each and proposes when to stop.
The harness checks the plan (round 1 must cover every query group), runs the tools, skips
repeated queries and makes the stop decision. A plan that still fails its schema after the one
repair retry is replaced by the preset query groups, so research never depends on the model
planning well.

Stop rule for now: 3 rounds, the model says done, or a round has nothing new to run. T6.11 adds
"< 2 new studies" and the coverage check before the model's `done` is accepted.
"""

import logging
from typing import Literal

from pydantic import ValidationError

from causal_genie.agent.state import (
    CausalQuery,
    FirstRoundPlan,
    FollowUpPlan,
    Query,
    QueryGroup,
    ResearchPlan,
    ResearchRound,
    RunState,
)
from causal_genie.llm.client import LLMClient
from causal_genie.tools.search import Searcher

log = logging.getLogger(__name__)

MAX_ROUNDS = 3


def research(
    state: RunState, llm: LLMClient, searcher: Searcher, max_rounds: int = MAX_ROUNDS
) -> None:
    """Run research rounds until the stop rule fires, appending each round to `state.rounds`."""
    ran: set[tuple[str, str]] = set()
    for number in range(1, max_rounds + 1):
        plan, planned_by = plan_round(state, llm, searcher, number, max_rounds)
        # Stored as a plain ResearchPlan: the round-specific checks already ran, and the saved
        # run must load back equal to the live one.
        stored = ResearchPlan.model_validate(plan.model_dump())
        state.rounds.append(ResearchRound(number=number, plan=stored, planned_by=planned_by))
        new = [query for query in plan.queries if (query.tool, query.text) not in ran]
        for query in new:
            run_query(query, number, searcher)
            ran.add((query.tool, query.text))
        skipped = len(plan.queries) - len(new)
        log.info(
            "round=%d planned_by=%s queries=%d skipped_repeats=%d done=%s done_reason=%r",
            number,
            planned_by,
            len(new),
            skipped,
            plan.done,
            plan.done_reason,
        )
        if plan.done or not new:
            log.info("research stops after round %d: %s", number, "done" if new else "no new")
            return
    log.info("research stops: reached %d rounds", max_rounds)


def plan_round(
    state: RunState, llm: LLMClient, searcher: Searcher, number: int, max_rounds: int
) -> tuple[ResearchPlan, Literal["model", "fallback"]]:
    """The model's plan for this round, or the preset plan if it fails its schema twice."""
    assert state.query is not None, "frame the question before research"
    schema = FirstRoundPlan if number == 1 else FollowUpPlan
    summary = summarize(state, searcher, number, max_rounds)
    try:
        return llm.complete("small", "research_plan", summary, schema), "model"
    except ValidationError as e:
        # PRD §5.9: after one repair retry, fall back to the fixed loop's preset query groups.
        log.warning("round=%d plan failed schema after repair, using fallback: %s", number, e)
        return fallback_plan(state.query), "fallback"


def run_query(query: Query, number: int, searcher: Searcher) -> None:
    """Run one planned query with the tool it names, logging the model's reason for it."""
    log.info(
        "round=%d tool=%s group=%s text=%r why=%r",
        number,
        query.tool,
        query.group,
        query.text,
        query.why,
    )
    if query.tool == "search":
        searcher.search(query.text)
    else:
        searcher.extract(query.text)


def fallback_plan(query: CausalQuery) -> FirstRoundPlan:
    """Preset query groups built from the PECO: one short query per group (the fixed loop)."""
    e, o = (v.name.replace("_", " ") for v in (query.exposure, query.outcome))
    preset: dict[QueryGroup, str] = {
        "best_answer": f"{e} {o} systematic review meta-analysis",
        "primary_studies": f"{e} {o} randomized controlled trial",
        "mechanism": f"{e} {o} mechanism",
        "confounders": f"{e} {o} confounding factors",
        "evidence_against": f"{e} {o} no effect",
        "data_availability": f"{e} {o} dataset",
    }
    return FirstRoundPlan(
        queries=[
            Query(text=text, tool="search", group=group, why="preset query group")
            for group, text in preset.items()
        ],
        done=True,
        done_reason="fallback: preset query groups only",
    )


def summarize(state: RunState, searcher: Searcher, number: int, max_rounds: int) -> str:
    """What the planner sees: the PECO, the round, queries already run and sources found."""
    assert state.query is not None
    q = state.query
    lines = [
        f"Question: {state.question}",
        "",
        "PECO:",
        f"- Population: {q.population}",
        f"- Exposure: {q.exposure.name} ({q.exposure.definition})",
        f"- Comparator: {q.comparator}",
        f"- Outcome: {q.outcome.name} ({q.outcome.definition})",
        f"- Time window: {q.time_window or 'not specified'}",
        "",
        f"Round {number} of {max_rounds} ({max_rounds - number} left after this one).",
        "",
        "Queries already run:",
    ]
    queries = [query for r in state.rounds for query in r.plan.queries]
    lines += [f"- [{x.tool} · {x.group}] {x.text}" for x in queries] or ["- (none)"]
    hits = {h.url: h for call in searcher.calls for h in call.hits}
    lines += ["", "Sources found so far (title — URL):"]
    lines += [f"- {h.title} — {h.url}" for h in hits.values()] or ["- (none)"]
    if searcher.extracts:
        lines += ["", "Full text fetched:"]
        lines += [f"- {x.url} ({'ok' if x.error is None else 'failed'})" for x in searcher.extracts]
    return "\n".join(lines)
