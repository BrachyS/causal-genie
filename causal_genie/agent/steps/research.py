"""Research loop (PRD §5.3): the model plans each round; the harness runs it and decides to stop.

The model writes the queries, picks `search` or `extract` for each and proposes when to stop.
The harness checks the plan (round 1 must cover every query group), runs the tools, skips
repeated queries, turns new sources into checked studies (`sources.process_sources`) and makes
the stop decision. A plan that still fails its schema after the one repair retry is replaced by
the preset query groups, so research never depends on the model planning well.

Stop rule: stop when a round has nothing new to run, after 3 rounds, when a round after the
first adds < 2 new studies, or when the model says done **and** no coverage is missing (a
later round whose plan says done with no gaps left is not run at all). A
`done` with coverage missing is overridden, and every summary names what is missing. "No
included study yet" is one of those gaps, so empty results get exactly one reformulated round
before the "< 2 new studies" rule ends the loop.
"""

import logging
from collections import Counter
from typing import Literal

from causal_genie.agent.certainty import certainty, design_tier
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
from causal_genie.agent.steps.sources import process_sources
from causal_genie.llm.client import LLMClient, ModelOutputError
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.search import RESEARCH_DOMAINS, Searcher

log = logging.getLogger(__name__)

MAX_ROUNDS = 3


def research(
    state: RunState,
    llm: LLMClient,
    searcher: Searcher,
    papers: EuropePMC,
    max_rounds: int = MAX_ROUNDS,
) -> None:
    """Run research rounds until the stop rule fires, appending each round to `state.rounds`
    and the studies and claims it finds to `state.studies` and `state.claims`."""
    for number in range(1, max_rounds + 1):
        if run_round(state, llm, searcher, papers, number, max_rounds).stop_reason:
            break
    state.certainty = certainty(state.studies)
    log.info("research done: certainty=%s", state.certainty)


def research_gaps(
    state: RunState,
    llm: LLMClient,
    searcher: Searcher,
    papers: EuropePMC,
    gaps: list[str],
    max_rounds: int = MAX_ROUNDS,
) -> None:
    """One more round, aimed at `gaps` found after research stopped (e.g. uncited DAG edges).

    The planner reads the latest round's `missing` list, so the gaps go there.
    """
    state.rounds[-1].missing += gaps
    run_round(state, llm, searcher, papers, len(state.rounds) + 1, max_rounds)
    state.certainty = certainty(state.studies)
    log.info("extra research done: gaps=%s certainty=%s", gaps, state.certainty)


def run_round(
    state: RunState,
    llm: LLMClient,
    searcher: Searcher,
    papers: EuropePMC,
    number: int,
    max_rounds: int,
) -> ResearchRound:
    """Plan and run one round, skipping queries earlier rounds already ran.

    A plan that says done when the last round left no gaps is not run: its queries would
    only add cost. Gaps from `research_gaps` sit in that `missing` list, so a gap round runs.
    """
    ran = {(query.tool, query.text) for r in state.rounds for query in r.plan.queries}
    gaps_before = state.rounds[-1].missing if state.rounds else None
    plan, planned_by = plan_round(state, llm, searcher, number, max_rounds)
    # Stored as a plain ResearchPlan: the round-specific checks already ran, and the saved
    # run must load back equal to the live one.
    stored = ResearchPlan.model_validate(plan.model_dump())
    current = ResearchRound(number=number, plan=stored, planned_by=planned_by)
    state.rounds.append(current)
    if plan.done and gaps_before == []:
        current.stop_reason = "model said done before running; coverage met"
        log.info("round=%d not run: %s (%s)", number, current.stop_reason, plan.done_reason)
        return current
    new = [query for query in plan.queries if (query.tool, query.text) not in ran]
    for query in new:
        run_query(query, number, searcher)
    before = included_ids(state)
    process_sources(state, llm, searcher, papers)
    current.new_studies = len(included_ids(state) - before)
    current.missing = missing_coverage(state)
    current.stop_reason = stop_reason(current, plan, bool(new), max_rounds)
    log.info(
        "round=%d planned_by=%s queries=%d skipped_repeats=%d new_studies=%d done=%s "
        "done_reason=%r missing=%s stop=%s",
        number,
        planned_by,
        len(new),
        len(plan.queries) - len(new),
        current.new_studies,
        plan.done,
        plan.done_reason,
        current.missing,
        current.stop_reason,
    )
    return current


def included_ids(state: RunState) -> set[str]:
    """Ids of the included studies so far."""
    return {s.id for s in state.studies if s.included}


def missing_coverage(state: RunState) -> list[str]:
    """Coverage the research still lacks; the model's `done` is accepted only when empty."""
    gaps = []
    if "evidence_against" not in {q.group for r in state.rounds for q in r.plan.queries}:
        gaps.append("no search for evidence against the leading answer has run yet")
    included = [s for s in state.studies if s.included]
    if not included:
        gaps.append(
            "no included study yet: reformulate the searches (synonyms, broader terms, "
            "technical names)"
        )
    # A news story or summary page is a gap only if no paper we read is at least as strong:
    # news of an RCT needs an RCT paper; a page summarizing reviews doesn't block a done.
    papers = [s for s in included if not s.secondary_only]
    strongest = max((design_tier(s) for s in papers), default=-1)
    gaps += [
        f"primary source for '{s.title}' (known only from {s.url})"
        for s in included
        if s.secondary_only and design_tier(s) > strongest
    ]
    return gaps


def stop_reason(
    current: ResearchRound, plan: ResearchPlan, ran_new: bool, max_rounds: int
) -> str | None:
    """Why research stops after this round, or None to keep going."""
    if not ran_new:
        return "no new queries to run"
    if current.number >= max_rounds:
        return f"reached {max_rounds} rounds"
    if current.number >= 2 and current.new_studies < 2:
        return f"round {current.number} added {current.new_studies} new studies (< 2)"
    if plan.done and not current.missing:
        return "model said done; coverage met"
    if plan.done:
        log.info("override: model said done but coverage is missing: %s", current.missing)
    return None


def plan_round(
    state: RunState, llm: LLMClient, searcher: Searcher, number: int, max_rounds: int
) -> tuple[ResearchPlan, Literal["model", "fallback"]]:
    """The model's plan for this round, or the preset plan if it fails its schema twice."""
    assert state.query is not None, "frame the question before research"
    schema = FirstRoundPlan if number == 1 else FollowUpPlan
    summary = summarize(state, searcher, number, max_rounds)
    try:
        return llm.complete("small", "research_plan", summary, schema), "model"
    except ModelOutputError as e:
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
        searcher.search(query.text, domains=RESEARCH_DOMAINS)
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
    """What the planner sees: the PECO, the round, queries already run, studies found so far
    (and which were read only from a snippet) and how many sources were excluded."""
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
    included = [s for s in state.studies if s.included]
    lines += ["", "Included studies (design, year, title — URL; how it was read; effect):"]
    lines += [
        f"- {s.design}, {s.year or 'year?'}, {s.title} — {s.url}; "
        f"{'full text' if s.full_text else 'snippet only'}"
        f"{'; news/secondary only' if s.secondary_only else ''}; {s.effect or 'no effect stated'}"
        for s in included
    ] or ["- (none)"]
    excluded = Counter(s.exclusion_reason for s in state.studies if not s.included)
    if excluded:
        reasons = ", ".join(f"{reason} {n}" for reason, n in excluded.most_common())
        lines += ["", f"Excluded sources: {sum(excluded.values())} ({reasons})"]
    if state.rounds and state.rounds[-1].missing:
        lines += ["", "Coverage still missing (target these this round):"]
        lines += [f"- {gap}" for gap in state.rounds[-1].missing]
    failed = [x.url for x in searcher.extracts if x.error is not None]
    if failed:
        lines += ["", "Full text could not be fetched (don't extract again):"]
        lines += [f"- {url}" for url in failed]
    return "\n".join(lines)
