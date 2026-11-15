"""Live runs from the app (T11.3): behind a passcode, a daily spend cap, one at a time.

Live mode is off unless `CG_LIVE_PASSCODE` is set (on Streamlit Cloud, a root-level secret
becomes an environment variable). The cap is checked before a run starts, so one run can
go past it by its own spend (~200–300k tokens, ~50 Tavily credits). The ledger is a JSON
file next to the live runs; on Streamlit Cloud it resets when the app restarts, so the
provider-side limits (Token Factory, Tavily) stay the hard stop.
"""

import hmac
import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tavily import TavilyClient

from causal_genie.agent.controller import CACHE_DIR, run
from causal_genie.agent.state import RunState
from causal_genie.llm.client import LLMClient, load_models, make_openai
from causal_genie.tools.download import Fetcher
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.search import Searcher

log = logging.getLogger(__name__)

RUN_LOCK = threading.Lock()  # one live run at a time across all browser sessions
LEDGER_LOCK = threading.Lock()


def passcode_ok(entered: str, expected: str) -> bool:
    """True only for the configured passcode; an empty one never matches (live mode off)."""
    return bool(expected) and hmac.compare_digest(entered.encode(), expected.encode())


def today() -> str:
    return datetime.now(UTC).date().isoformat()


class SpendLedger:
    """Live runs and model tokens spent per UTC day, kept in a JSON file."""

    def __init__(self, path: Path, max_runs: int, max_tokens: int) -> None:
        self.path, self.max_runs, self.max_tokens = path, max_runs, max_tokens

    def _read(self) -> dict[str, dict[str, int]]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def refusal(self, day: str) -> str | None:
        """Why no live run may start on `day`, or None if one may."""
        spent = self._read().get(day, {"runs": 0, "tokens": 0})
        if spent["runs"] >= self.max_runs:
            return f"Daily limit of {self.max_runs} live runs reached; try again tomorrow."
        if spent["tokens"] >= self.max_tokens:
            return (
                f"Daily token limit reached ({spent['tokens']:,} of {self.max_tokens:,}); "
                "try again tomorrow."
            )
        return None

    def record(self, day: str, tokens: int, runs: int = 1) -> None:
        """Add `runs` live runs and `tokens` model tokens to `day`."""
        with LEDGER_LOCK:
            ledger = self._read()
            spent = ledger.setdefault(day, {"runs": 0, "tokens": 0})
            spent["runs"] += runs
            spent["tokens"] += tokens
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(ledger, indent=2))
        log.info("live spend %s: %s", day, spent)


def live_dir() -> Path:
    """Where live runs and the spend ledger are saved (never the committed gallery)."""
    return Path(os.getenv("CG_LIVE_DIR", "runs"))


def ledger() -> SpendLedger:
    return SpendLedger(
        live_dir() / "live_spend.json",
        max_runs=int(os.getenv("CG_LIVE_RUNS_PER_DAY", "5")),
        max_tokens=int(os.getenv("CG_LIVE_TOKENS_PER_DAY", "1500000")),
    )


def run_live(question: str, runs_dir: Path) -> RunState:
    """Answer `question` with the real model, search and data step (no clarifying
    questions: the framing's ambiguities are listed in the report)."""
    load_dotenv()
    genie = os.getenv("GENIE_MODE", "").lower() in ("1", "true", "yes", "on")
    llm = LLMClient(make_openai(), load_models())
    searcher = Searcher(TavilyClient(api_key=os.environ["TAVILY_API_KEY"]))
    with httpx.Client(timeout=60) as http:
        return run(
            question,
            llm,
            searcher,
            EuropePMC(http),
            Fetcher(http),
            runs_dir,
            CACHE_DIR,
            ask=None,
            genie=genie,
        )
