"""Live mode in the app (T11.3): passcode, daily spend cap, one run at a time.

No real run here: `run_live` is replaced, so these tests spend no tokens.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from causal_genie.agent.state import RunState
from causal_genie.ui import live
from causal_genie.ui.live import SpendLedger, passcode_ok

APP = Path(__file__).parents[1] / "causal_genie" / "ui" / "app.py"
NEW_QUESTION = "Does coffee cause gout?"


def test_passcode_ok() -> None:
    assert passcode_ok("open sesame", "open sesame")
    assert not passcode_ok("open", "open sesame")
    assert not passcode_ok("", "")  # no passcode configured: live mode is off


def test_ledger_counts_runs_and_tokens_per_day(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "spend.json", max_runs=3, max_tokens=1000)
    assert ledger.refusal("2026-11-14") is None

    ledger.record("2026-11-14", tokens=600)
    assert ledger.refusal("2026-11-14") is None
    ledger.record("2026-11-14", tokens=500)

    assert "token" in (ledger.refusal("2026-11-14") or "")
    assert ledger.refusal("2026-11-15") is None  # a new day starts fresh
    # the ledger lives on disk, so another session (or a restart) sees the same spend
    assert SpendLedger(tmp_path / "spend.json", 3, 1000).refusal("2026-11-14") is not None


def test_ledger_caps_runs_per_day(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "spend.json", max_runs=1, max_tokens=10**9)
    ledger.record("2026-11-14", tokens=1)
    assert "run" in (ledger.refusal("2026-11-14") or "")


@pytest.fixture
def live_app(sample_state, tmp_path, monkeypatch) -> list[str]:
    """The app with live mode on and a fake run; returns the questions run live."""
    sample_state.report = "# report"
    sample_state.verdict = "published_only"
    sample_state.save(tmp_path / "gallery")
    monkeypatch.setenv("CG_RUNS_DIR", str(tmp_path / "gallery"))
    monkeypatch.setenv("CG_LIVE_PASSCODE", "sesame")
    monkeypatch.setenv("CG_LIVE_DIR", str(tmp_path / "live"))
    monkeypatch.setenv("CG_LIVE_RUNS_PER_DAY", "1")
    ran: list[str] = []

    def fake_run(question: str, runs_dir: Path) -> RunState:
        ran.append(question)
        return sample_state.model_copy(update={"question": question, "run_id": "live1"})

    monkeypatch.setattr(live, "run_live", fake_run)
    return ran


def ask(question: str, passcode: str) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.text_input[0].input(question).run()
    at.text_input[1].input(passcode)
    at.button[0].click().run()
    return at


def test_live_run_needs_the_passcode(live_app) -> None:
    at = ask(NEW_QUESTION, "wrong")

    assert not at.exception
    assert any("passcode" in m.value.lower() for m in at.error)
    assert live_app == []


def test_live_run_with_passcode_shows_the_run_and_counts_it(live_app) -> None:
    at = ask(NEW_QUESTION, "sesame")

    assert not at.exception
    assert live_app == [NEW_QUESTION]
    assert any("Published evidence only" in m.value for m in at.markdown)

    again = ask("Does tea cause gout?", "sesame")  # cap is 1 run per day
    assert any("limit" in m.value.lower() for m in again.warning)
    assert live_app == [NEW_QUESTION]


def test_live_mode_off_without_a_configured_passcode(live_app, monkeypatch) -> None:
    monkeypatch.delenv("CG_LIVE_PASSCODE")
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.text_input[0].input(NEW_QUESTION).run()

    assert not at.exception
    assert len(at.text_input) == 1 and not at.button
    assert any("Live runs are off" in m.value for m in at.info)
