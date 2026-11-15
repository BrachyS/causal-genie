"""Copy saved runs into `gallery/`, the recorded runs the public demo shows (T11.2).

`runs/` is never committed; the gallery is. The copy keeps everything the app displays
(verdict, prose, studies, claims and their verbatim quotes, DAG, results, the queries run,
token counts) and blanks the raw third-party text a run records for tracing: search
snippets, full page text, and the model calls' inputs and outputs. So a gallery run is for
showing, not for `--replay`, which needs the full run in `runs/`.

Usage: `uv run python -m causal_genie.ui.gallery RUN_ID [RUN_ID ...] [--genie]`

`--genie` first adds the Genie's aside to a full run in `runs/` that was recorded without
Genie mode (one small-model call; the gate still decides), so the gallery copy shows it.
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from causal_genie.agent.state import RunState
from causal_genie.agent.steps.genie import genie_aside
from causal_genie.agent.steps.report import write_report
from causal_genie.llm.client import LLMClient, load_models, make_openai
from causal_genie.log import setup_logging

log = logging.getLogger(__name__)

GALLERY_DIR = Path("gallery")


def export(run_dir: Path, gallery_dir: Path) -> Path:
    """Write the display copy of the run in `run_dir`; returns its `state.json` path."""
    state = RunState.load(run_dir / "state.json")
    for search in state.searches:
        for hit in search.hits:
            hit.content = ""
    for extract in state.extracts:
        extract.content = None
    for call in state.llm_calls:
        call.user_content = call.response = ""
    path = state.save(gallery_dir)
    log.info("exported run %s (%r) to %s", state.run_id, state.question, path)
    return path


def add_genie(run_dir: Path, llm: LLMClient) -> RunState:
    """Gate the run in `run_dir` for a Genie aside if it never was, and save it with the
    `genie_gate` call recorded, so `--replay` reproduces the aside."""
    path = run_dir / "state.json"
    state = RunState.load(path)
    if any(c.prompt_name == "genie_gate" for c in state.llm_calls):
        log.info("run %s already went through the Genie gate", state.run_id)
        return state
    state.genie = genie_aside(state, llm)
    state.llm_calls += llm.calls
    state.report = write_report(state)
    state.save(run_dir.parent)
    (run_dir / "report.md").write_text(state.report)
    log.info("run %s: genie aside %s", state.run_id, "added" if state.genie else "gated off")
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Copy saved runs into the demo gallery.")
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--to", type=Path, default=GALLERY_DIR)
    parser.add_argument("--genie", action="store_true", help="add the Genie aside first")
    args = parser.parse_args(argv)
    setup_logging()
    for run_id in args.run_ids:
        if args.genie:
            load_dotenv()
            add_genie(args.runs / run_id, LLMClient(make_openai(), load_models()))
        export(args.runs / run_id, args.to)


if __name__ == "__main__":
    main()
