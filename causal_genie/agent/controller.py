"""Run one question end to end and save everything needed to inspect or replay it.

The controller, never a model, picks the verdict. Each run is saved to `runs/<run_id>/` as
`state.json` (every model call, search, download and result) plus `report.md` and the DAG
(`dag.json`, `dag.png`), even when a step fails, so failures can be debugged from the saved
state. Rendering the DAG needs Graphviz's `dot` program installed.

Usage: `uv run python -m causal_genie.agent.controller "Does X cause Y?" [--replay]`
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from tavily import TavilyClient

from causal_genie.agent.router import normalize, route
from causal_genie.agent.state import (
    Adjustment,
    DatasetCandidate,
    NodeMapping,
    RunState,
    Study,
    Verdict,
)
from causal_genie.agent.steps.dag import run_dag
from causal_genie.agent.steps.data import find_data
from causal_genie.agent.steps.estimate import estimate_backdoor, estimate_rct
from causal_genie.agent.steps.frame import frame
from causal_genie.agent.steps.identify import map_columns, source
from causal_genie.agent.steps.report import write_report
from causal_genie.agent.steps.research import research
from causal_genie.llm.client import LLMClient, load_models, make_openai
from causal_genie.log import setup_logging
from causal_genie.tools.derive import node_table
from causal_genie.tools.identify import detect_design, identify, precision_set
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.profile import read_table
from causal_genie.tools.render import render_dag
from causal_genie.tools.search import Searcher

log = logging.getLogger(__name__)

RUNS_DIR = Path("runs")
CACHE_DIR = Path("cache") / "downloads"


def new_run_id() -> str:
    """Sortable by time, unique across concurrent runs."""
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"


def run(
    question: str,
    llm: LLMClient,
    searcher: Searcher,
    papers: EuropePMC,
    http: httpx.Client,
    runs_dir: Path,
    cache_dir: Path,
    ask: Callable[[str], str] | None,
) -> RunState:
    """Route → frame → research → DAG → data → map → design → estimate → verdict → report,
    saving the run."""
    state = RunState(
        run_id=new_run_id(),
        question=question,
        versions={"causal_genie": version("causal-genie")}
        | {f"model_{tier}": model for tier, model in llm.models.items()},
    )

    def ask_and_record(clarifying: str) -> str:
        """Ask the user and keep the answer on the run, so replay can reproduce it."""
        assert ask is not None
        state.clarifications[clarifying] = ask(clarifying)
        return state.clarifications[clarifying]

    try:
        state.route = route(question, llm, cached={}, gallery={})
        if state.route.kind == "not_causal":
            state.verdict = "not_causal"
        else:
            state.query = frame(question, llm, ask_and_record if ask else None)
            research(state, llm, searcher, papers)
            if any(s.included for s in state.studies):
                run_dag(state, llm, searcher, papers)
                render_dag(state.edges, runs_dir / state.run_id)
            state.verdict = analyze(
                state, llm, searcher, http, cache_dir, ask_and_record if ask else None
            )
        record_calls(state, llm, searcher, papers)  # the report lists the sources found
        state.report = write_report(state)
    finally:
        record_calls(state, llm, searcher, papers)
        path = state.save(runs_dir)
        if state.report is not None:
            (path.parent / "report.md").write_text(state.report)
        log.info(
            "run_id=%s verdict=%s saved=%s tokens=%s",
            state.run_id,
            state.verdict,
            path,
            state.tokens,
        )
    return state


def analyze(
    state: RunState,
    llm: LLMClient,
    searcher: Searcher,
    http: httpx.Client,
    cache_dir: Path,
    ask: Callable[[str], str] | None,
) -> Verdict:
    """Find data, map it to the DAG, label its design and estimate if the design allows;
    returns the verdict (PRD §5.2). Every decision here is a rule, never a model's call."""
    assert state.query is not None
    exposure, outcome = state.query.exposure.name, state.query.outcome.name
    no_estimate: Verdict = (
        "published_only" if any(s.included for s in state.studies) else "insufficient"
    )
    accepted = find_data(state, llm, searcher, http, cache_dir)
    if accepted is None:
        restricted = any(c.access == "restricted" for c in state.datasets)
        return "restricted" if restricted else no_estimate
    mapping = map_columns(state, llm, ask)
    by_node = {m.node: m for m in mapping.nodes if m.columns} if mapping else {}
    if exposure not in by_node or outcome not in by_node:
        log.info("no estimate: exposure or outcome not mapped (mapping=%s)", mapping)
        return no_estimate
    assert mapping is not None
    data = state.downloads[-1]
    table = node_table(read_table(Path(data.path)), mapping)
    trial = source_trial(state, accepted)
    randomized = trial is not None or user_says_randomized(by_node[exposure], accepted, ask)
    roster = trial.n if trial is not None and documented(state, f"{trial.id}-n") else None
    state.design = detect_design(
        table,
        exposure,
        outcome,
        state.edges,
        randomized,
        roster=roster,
        unit=mapping.unit,
        time=mapping.time,
    )
    if not state.design.estimable:
        return "study_plan"
    binary = set(table[outcome].dropna().unique()) <= {0, 1}
    scale = "risk_difference" if binary else "mean_difference"
    if state.design.design == "randomized":
        state.adjustment = Adjustment(adjust_for=[], missing=[], randomized=True)
        mapped_from = {n: source(by_node[n]) for n in (exposure, outcome)}
        state.results.append(
            estimate_rct(table, data.sha256, exposure, outcome, scale, mapped_from)
        )
        return "estimated"
    observed = [c for c in table.columns if c not in (mapping.unit, mapping.time)]
    adjustment = identify(state.edges, exposure, outcome, observed, randomized=False)
    assert adjustment.adjust_for is not None, "detect_design said estimable"
    precision = precision_set(state.edges, exposure, outcome, adjustment.adjust_for, observed)
    state.adjustment = adjustment.model_copy(update={"precision": precision})
    if binary and state.query.effect_scale == "odds_ratio":
        scale = "odds_ratio"
    used = [exposure, outcome, *adjustment.adjust_for, *precision]
    state.results.append(
        estimate_backdoor(
            table,
            data.sha256,
            exposure,
            outcome,
            adjustment.adjust_for,
            scale,
            precision,
            {n: source(by_node[n]) for n in used},
        )
    )
    return "estimated"


def documented(state: RunState, claim_id: str) -> bool:
    """The ledger claim exists and passed both citation checks."""
    return any(c.id == claim_id and c.status == "documented" for c in state.claims)


def source_trial(state: RunState, accepted: DatasetCandidate) -> Study | None:
    """The included RCT whose own data link gave the accepted table, if its design is
    documented (quote found and supported): then assignment is documented by the paper."""
    for s in state.studies:
        if s.id == accepted.study_id and s.included and s.design == "RCT":
            if documented(state, f"{s.id}-design"):
                return s
            log.info("study %s is an RCT, but its design claim is not documented", s.id)
    return None


def user_says_randomized(
    exposure: NodeMapping, accepted: DatasetCandidate, ask: Callable[[str], str] | None
) -> bool:
    """Ask the user whether assignment was random; only for an exposure that is a column as
    is (a derived one, like moon phase, is never assigned). No user → not randomized."""
    if ask is None or exposure.derive is not None:
        return False
    reply = ask(
        f"Was {exposure.node} (column {exposure.columns[0]}) randomly assigned in "
        f"{accepted.title}, as in a trial or A/B test? (yes/no)"
    )
    return reply.strip().lower().startswith("y")


def record_calls(state: RunState, llm: LLMClient, searcher: Searcher, papers: EuropePMC) -> None:
    """Copy the model calls, searches, extracts and lookups made so far into the run state."""
    state.llm_calls = list(llm.calls)
    state.searches = list(searcher.calls)
    state.extracts = list(searcher.extracts)
    state.lookups = list(papers.calls)


def find_run(question: str, runs_dir: Path) -> Path | None:
    """Latest saved run of the same question (ignoring case and punctuation), for replay.

    Reads only each run's question, so runs saved under an older schema don't block the
    search; the run that is replayed is still fully validated when loaded.
    """
    key = normalize(question)
    matches = [
        path
        for path in sorted(runs_dir.glob("*/state.json"))
        if normalize(json.loads(path.read_text())["question"]) == key
    ]
    return matches[-1] if matches else None


def ask_user(clarifying: str) -> str:
    """Ask the user a clarifying question on the terminal and return their answer."""
    return input(f"{clarifying}\n> ")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: answer one question live, or replay its latest saved run."""
    parser = argparse.ArgumentParser(description="Answer a 'does X cause Y?' question.")
    parser.add_argument("question")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="replay the latest saved run of this question; no keys needed (or set CG_REPLAY=1)",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    setup_logging()

    ask: Callable[[str], str] | None = None
    replaying = args.replay or os.getenv("CG_REPLAY") == "1"
    if replaying:
        path = find_run(args.question, RUNS_DIR)
        if path is None:
            raise SystemExit(f"no saved run of {args.question!r} in {RUNS_DIR}/ to replay")
        recorded = RunState.load(path)
        llm = LLMClient.from_replay(recorded.llm_calls)
        searcher = Searcher.from_replay(recorded.searches, recorded.extracts)
        papers = EuropePMC.from_replay(recorded.lookups)
        if recorded.clarifications:
            ask = recorded.clarifications.__getitem__
    else:
        llm = LLMClient(make_openai(), load_models())
        searcher = Searcher(TavilyClient(api_key=os.environ["TAVILY_API_KEY"]))
        if sys.stdin.isatty():
            ask = ask_user

    with httpx.Client(timeout=60) as http:
        if not replaying:
            papers = EuropePMC(http)
        state = run(args.question, llm, searcher, papers, http, RUNS_DIR, CACHE_DIR, ask)
    sys.stdout.write(state.report or "")


if __name__ == "__main__":
    main()
