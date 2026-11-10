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
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from tavily import TavilyClient

from causal_genie.agent.router import normalize, route
from causal_genie.agent.state import RunState
from causal_genie.agent.steps.dag import run_dag
from causal_genie.agent.steps.estimate import estimate_rct
from causal_genie.agent.steps.frame import frame
from causal_genie.agent.steps.report import write_report
from causal_genie.agent.steps.research import research
from causal_genie.llm.client import LLMClient, load_models, make_openai
from causal_genie.log import setup_logging
from causal_genie.tools.download import download
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.render import render_dag
from causal_genie.tools.search import Searcher

log = logging.getLogger(__name__)

RUNS_DIR = Path("runs")
CACHE_DIR = Path("cache") / "downloads"


@dataclass(frozen=True)
class TrialData:
    """Where a randomized trial's table lives and which columns hold the PECO variables."""

    url: str
    license: str
    doi: str | None
    treatment: str  # 0/1, 1 = exposure
    outcome: str
    cutoff: float | None  # binary outcome = outcome > cutoff; None keeps the outcome as is
    scale: Literal["risk_difference", "mean_difference"]


# T5 thin slice: D1's data and column mapping are fixed here until data discovery (T8) and
# mapping (T9.1) find them. Keyed by the exact question, so no other question can use this data.
THIN_SLICE_DATA = {
    normalize("Does licorice gargling reduce post-op sore throat?"): TrialData(
        url="https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/licorice_gargle.csv",
        license="MIT",
        doi=None,
        treatment="treat",
        outcome="pacu30min_throatPain",  # 0–10 pain score, 30 min after surgery
        cutoff=0,  # any sore throat
        scale="risk_difference",
    )
}


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
    """Route → frame → research → DAG → download → estimate → verdict → report, saving the run."""
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
            data = THIN_SLICE_DATA.get(normalize(question))
            if data is None and not any(s.included for s in state.studies):
                # PRD §5.9: nothing useful found after one reformulated round → ❓
                state.verdict = "insufficient"
            elif data is None:
                # Thin slice: no data discovery yet (T8); the verdict rules (T10.1) refine this.
                state.verdict = "published_only"
            else:
                d = download(data.url, http, cache_dir, doi=data.doi, license=data.license)
                state.downloads.append(d)
                state.results.append(
                    estimate_rct(
                        Path(d.path),
                        d.sha256,
                        data.treatment,
                        data.outcome,
                        data.cutoff,
                        data.scale,
                    )
                )
                state.verdict = "estimated"
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
