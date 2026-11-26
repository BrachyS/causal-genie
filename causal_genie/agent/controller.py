"""Run one question end to end and save everything needed to inspect or replay it.

The controller, never a model, picks the verdict. Each run is saved to `runs/<run_id>/` as
`state.json` (every model call, search, download and result) plus `report.md` and the DAG
(`dag.json`, `dag.png`), even when a step fails, so failures can be debugged from the saved
state. When a model, search or HTTP API still fails after its retry, the run stops there and
the report is partial, naming the failed step (PRD §5.9). Rendering the DAG needs Graphviz's
`dot` program installed.

Usage: `uv run python -m causal_genie.agent.controller "Does X cause Y?" [--replay]
[--data TABLE.csv]`; set `GENIE_MODE=1` for the Genie's aside.
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
import openai
import requests
import tavily.errors
from dotenv import load_dotenv
from tavily import TavilyClient

from causal_genie.agent.router import normalize, route
from causal_genie.agent.state import (
    Adjustment,
    DatasetCandidate,
    NodeMapping,
    RunState,
    Study,
)
from causal_genie.agent.steps.dag import run_dag
from causal_genie.agent.steps.data import find_data, use_table
from causal_genie.agent.steps.estimate import cluster_column, estimate_backdoor, estimate_rct
from causal_genie.agent.steps.frame import frame
from causal_genie.agent.steps.genie import genie_aside
from causal_genie.agent.steps.identify import map_columns, source
from causal_genie.agent.steps.report import draft_report, write_report
from causal_genie.agent.steps.research import research
from causal_genie.agent.verdict import pick_verdict
from causal_genie.llm.client import LLMClient, ModelOutputError, load_models, make_openai
from causal_genie.log import setup_logging
from causal_genie.tools.derive import node_table
from causal_genie.tools.download import Fetcher
from causal_genie.tools.identify import detect_design, identify, precision_set
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.profile import read_table
from causal_genie.tools.render import render_dag
from causal_genie.tools.repositories import DryadAuth
from causal_genie.tools.search import Searcher

log = logging.getLogger(__name__)

RUNS_DIR = Path("runs")
CACHE_DIR = Path("cache") / "downloads"
# What stops a run with a partial report naming the step: an external API still failing
# after its retry, or a model's output still failing its schema after the repair retry.
STEP_FAILURES = (
    ModelOutputError,
    openai.APIError,
    httpx.HTTPError,
    requests.RequestException,  # Tavily's transport
    tavily.errors.TimeoutError,
    tavily.errors.UsageLimitExceededError,
)


def new_run_id() -> str:
    """Sortable by time, unique across concurrent runs."""
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"


def run(
    question: str,
    llm: LLMClient,
    searcher: Searcher,
    papers: EuropePMC,
    fetcher: Fetcher,
    runs_dir: Path,
    cache_dir: Path,
    ask: Callable[[str], str] | None,
    data: Path | None = None,
    genie: bool = False,
) -> RunState:
    """Route → frame → research → DAG → data → map → design → estimate → verdict → report,
    saving the run.

    `data` is the user's own table: it replaces the data search. `genie` turns on the
    Genie's aside, still gated per run.
    """
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
        try:
            with state.timed("route"):
                state.route = route(question, llm, cached={}, gallery={})
            if state.route.kind == "causal":
                with state.timed("frame"):
                    state.query = frame(question, llm, ask_and_record if ask else None)
                with state.timed("research"):
                    research(state, llm, searcher, papers)
                if any(s.included for s in state.studies):
                    with state.timed("dag"):
                        run_dag(state, llm, searcher, papers)
                        render_dag(state.edges, runs_dir / state.run_id)
                analyze(
                    state, llm, searcher, fetcher, cache_dir, ask_and_record if ask else None, data
                )
            state.verdict = pick_verdict(state)
            if state.verdict != "not_causal":
                with state.timed("report"):
                    state.prose = draft_report(state, llm)
            if genie:
                with state.timed("genie"):
                    state.genie = genie_aside(state, llm)
            state.step = None
        except STEP_FAILURES as e:
            log.exception(
                "run_id=%s: step %s failed; writing a partial report", state.run_id, state.step
            )
            state.failure = f"{type(e).__name__}: {e}"
        record_calls(state, llm, searcher, papers, fetcher)  # the report lists the sources found
        state.report = write_report(state)
    finally:
        record_calls(state, llm, searcher, papers, fetcher)
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
    fetcher: Fetcher,
    cache_dir: Path,
    ask: Callable[[str], str] | None,
    data: Path | None = None,
) -> None:
    """Find data (or use the user's `data`), map it to the DAG, label its design and
    estimate if the design allows. Results land on `state`; `pick_verdict` reads them.
    Every decision here is a rule, never a model's call."""
    assert state.query is not None
    exposure, outcome = state.query.exposure.name, state.query.outcome.name
    with state.timed("data"):
        if data is not None:
            accepted = use_table(state, data)
        else:
            accepted = find_data(state, llm, searcher, fetcher, cache_dir)
    if accepted is None:
        return
    with state.timed("map"):
        mapping = map_columns(state, llm, ask)
    by_node = {m.node: m for m in mapping.nodes if m.columns} if mapping else {}
    if exposure not in by_node or outcome not in by_node:
        log.info("no estimate: exposure or outcome not mapped (mapping=%s)", mapping)
        return
    assert mapping is not None
    with state.timed("estimate"):
        table_file = state.downloads[-1]
        table = node_table(read_table(Path(table_file.path)), mapping)
        # A node with no values at all (e.g. a date rule applied to a weekday-name column)
        # would drop every row: leave it out, as if unmapped.
        empty = [node for node in by_node if table[node].isna().all()]
        if empty:
            log.warning("mapped nodes with no values, left out as unmapped: %s", empty)
            table = table.drop(columns=empty)
        if exposure in empty or outcome in empty:
            return
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
            return
        binary = set(table[outcome].dropna().unique()) <= {0, 1}
        scale = "risk_difference" if binary else "mean_difference"
        if state.design.design == "randomized":
            state.adjustment = Adjustment(adjust_for=[], missing=[], randomized=True)
            mapped_from = {n: source(by_node[n]) for n in (exposure, outcome)}
            state.results.append(
                estimate_rct(table, table_file.sha256, exposure, outcome, scale, mapped_from)
            )
            return
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
                table_file.sha256,
                exposure,
                outcome,
                adjustment.adjust_for,
                scale,
                precision,
                {n: source(by_node[n]) for n in used},
                cluster_column(table, mapping.time),
            )
        )


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


def record_calls(
    state: RunState, llm: LLMClient, searcher: Searcher, papers: EuropePMC, fetcher: Fetcher
) -> None:
    """Copy the model calls, searches, extracts, lookups and data fetches made so far into
    the run state."""
    state.llm_calls = list(llm.calls)
    state.searches = list(searcher.calls)
    state.extracts = list(searcher.extracts)
    state.lookups = list(papers.calls)
    state.fetches = list(fetcher.calls)


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
        "--data",
        type=Path,
        help="your own table (CSV, TSV, XLSX or Parquet) to analyze instead of searching",
    )
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
    genie = os.getenv("GENIE_MODE", "").lower() in ("1", "true", "yes", "on")
    if replaying:
        path = find_run(args.question, RUNS_DIR)
        if path is None:
            raise SystemExit(f"no saved run of {args.question!r} in {RUNS_DIR}/ to replay")
        recorded = RunState.load(path)
        llm = LLMClient.from_replay(recorded.llm_calls)
        if recorded.clarifications:
            ask = recorded.clarifications.__getitem__
        genie = any(c.prompt_name == "genie_gate" for c in recorded.llm_calls)

        state = run(
            args.question,
            llm,
            Searcher.from_replay(recorded.searches, recorded.extracts),
            EuropePMC.from_replay(recorded.lookups),
            Fetcher.from_replay(recorded.fetches),
            RUNS_DIR,
            CACHE_DIR,
            ask,
            args.data,
            genie,
        )
    else:
        llm = LLMClient(make_openai(), load_models())
        searcher = Searcher(TavilyClient(api_key=os.environ["TAVILY_API_KEY"]))
        if sys.stdin.isatty():
            ask = ask_user
        with httpx.Client(timeout=60) as http:
            state = run(
                args.question,
                llm,
                searcher,
                EuropePMC(http),
                Fetcher(http, dryad=DryadAuth.from_env()),
                RUNS_DIR,
                CACHE_DIR,
                ask,
                args.data,
                genie,
            )
    sys.stdout.write(state.report or "")


if __name__ == "__main__":
    main()
