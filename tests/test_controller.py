import json
from pathlib import Path
from typing import cast

import httpx
import pytest
from conftest import epmc_result, extraction, proposal
from tavily import TavilyClient

from causal_genie.agent import controller
from causal_genie.agent.state import RunState
from causal_genie.llm.client import LLMClient
from causal_genie.tools import download as dl
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.search import Searcher

D1 = "Does licorice gargling reduce post-op sore throat?"
# 3/4 vs 1/3 with sore throat (pain > 0); one row has no outcome
CSV = (
    b"rownames,treat,pacu30min_throatPain\n1,0,3\n2,0,1\n3,0,0\n4,0,2\n5,1,0\n6,1,4\n7,1,0\n8,1,\n"
)


READ = extraction(quotes=[])  # the fake Tavily's one page, read as the D1 trial
# READ yields no claims, so the DAG's edge is assumed and research is asked for one more round.
DAG = proposal(("licorice_gargle", "sore_throat", []))
NO_QUERIES = json.dumps({"queries": [], "done": True, "done_reason": "test"})


def routed(causal: bool) -> str:
    """Router model output saying whether the question is causal."""
    return json.dumps({"causal": causal, "reason": "test", "causal_rewrite": None})


@pytest.fixture
def framed(sample_state: RunState) -> str:
    """Frame model output: the sample run's estimand as JSON."""
    assert sample_state.query is not None
    return sample_state.query.model_dump_json()


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> httpx.Client:
    """HTTP client that serves `CSV` for any URL, with DNS faked to a public IP."""
    monkeypatch.setattr(dl, "resolve_ips", lambda host: ["185.199.108.153"])
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=CSV)))


@pytest.fixture
def run_live(make_llm, make_epmc, fake_tavily, http, tmp_path: Path):
    """Run the controller end to end on fakes, given the model outputs in order."""

    def run(question: str, outcomes: list) -> RunState:
        llm, _ = make_llm(outcomes)
        searcher = Searcher(cast(TavilyClient, fake_tavily))
        papers, _ = make_epmc([epmc_result()])
        return controller.run(
            question, llm, searcher, papers, http, tmp_path / "runs", tmp_path / "cache", ask=None
        )

    return run


def test_d1_runs_end_to_end_and_saves_the_run(run_live, framed, round_plan, tmp_path: Path) -> None:
    state = run_live(D1, [routed(True), framed, round_plan, READ, DAG, NO_QUERIES])

    assert state.verdict == "estimated"
    (result,) = state.results
    out = result.outputs
    assert (out["events_treated"], out["n_treated"], out["events_control"], out["n_control"]) == (
        1,
        3,
        3,
        4,
    )
    assert out["rows_excluded_missing"] == 1
    assert len(state.searches) == 6  # one per query group
    assert [r.planned_by for r in state.rounds] == ["model", "model"]
    assert [(e.source, e.target, e.assumed) for e in state.edges] == [
        ("licorice_gargle", "sore_throat", True)
    ]
    assert state.downloads[0].license == "MIT"
    assert [c.prompt_name for c in state.llm_calls] == [
        "route",
        "frame",
        "research_plan",
        "extract",
        "dag",
        "research_plan",
    ]
    assert state.report is not None and "✅" in state.report
    assert "## Evidence" in state.report
    run_dir = tmp_path / "runs" / state.run_id
    assert RunState.load(run_dir / "state.json") == state
    assert (run_dir / "report.md").read_text() == state.report
    assert json.loads((run_dir / "dag.json").read_text())["edges"][0]["assumed"] is True
    assert (run_dir / "dag.png").read_bytes().startswith(b"\x89PNG")


def test_not_causal_question_stops_after_routing(run_live, fake_tavily) -> None:
    state = run_live("How many babies are born each day?", [routed(False)])

    assert state.verdict == "not_causal"
    assert fake_tavily.requests == []
    assert state.query is None


def test_question_without_thin_slice_data_is_not_estimated(run_live, framed, round_plan) -> None:
    knuckles = "Does cracking your knuckles cause arthritis?"
    state = run_live(knuckles, [routed(True), framed, round_plan, READ, DAG, NO_QUERIES])

    assert state.verdict == "published_only"
    assert state.downloads == [] and state.results == []
    assert len(state.searches) == 6


def test_no_included_study_and_no_data_is_insufficient(run_live, framed, round_plan) -> None:
    knuckles = "Does cracking your knuckles cause arthritis?"
    off_topic = extraction(included=False, quotes=[])
    round_2 = json.dumps(
        {
            "queries": [
                {
                    "text": "knuckle cracking osteoarthritis",
                    "tool": "search",
                    "group": "primary_studies",
                    "why": "reformulated",
                }
            ],
            "done": True,
            "done_reason": "test",
        }
    )

    state = run_live(knuckles, [routed(True), framed, round_plan, off_topic, round_2])

    assert state.verdict == "insufficient"
    assert state.edges == []  # no evidence to build a graph from
    assert state.report is not None and "No included studies" in state.report


def test_failed_step_still_saves_the_run(run_live, tmp_path: Path) -> None:
    with pytest.raises(IndexError):  # the fake model has no answer for the frame call
        run_live(D1, [routed(True)])

    (saved,) = (tmp_path / "runs").glob("*/state.json")
    state = RunState.load(saved)
    assert state.verdict is None
    assert [c.prompt_name for c in state.llm_calls] == ["route"]


def test_replay_reproduces_a_run_without_model_or_search(
    run_live, framed, round_plan, http, prompts_dir: Path, tmp_path: Path
) -> None:
    live = run_live(D1, [routed(True), framed, round_plan, READ, DAG, NO_QUERIES])

    replayed = controller.run(
        D1,
        LLMClient.from_replay(live.llm_calls, prompts_dir=prompts_dir),
        Searcher.from_replay(live.searches, live.extracts),
        EuropePMC.from_replay(live.lookups),
        http,
        tmp_path / "runs",
        tmp_path / "cache",
        ask=None,
    )

    assert replayed.results == live.results
    assert replayed.rounds == live.rounds
    assert replayed.edges == live.edges
    assert replayed.report is not None and live.report is not None

    def stable(state: RunState) -> str:
        """The report minus what legitimately differs: run id and download time (replay still
        downloads the data again until T11.4, so a second can tick between the two runs)."""
        assert state.report is not None
        text = state.report.replace(state.run_id, "")
        for d in state.downloads:
            text = text.replace(d.retrieved_at, "")
        return text

    assert stable(replayed) == stable(live)


def test_find_run_picks_latest_run_of_the_same_question(tmp_path: Path) -> None:
    for run_id in ["20261101-090000-aaaaaa", "20261102-090000-bbbbbb"]:
        RunState(run_id=run_id, question=D1).save(tmp_path)
    RunState(run_id="20261103-090000-cccccc", question="Another question?").save(tmp_path)
    old = tmp_path / "20261031-090000-dddddd" / "state.json"  # saved under an older schema
    old.parent.mkdir()
    old.write_text(json.dumps({"run_id": "x", "question": "Old?", "plans": []}))

    assert controller.find_run("does licorice gargling reduce post-op sore throat", tmp_path) == (
        tmp_path / "20261102-090000-bbbbbb" / "state.json"
    )
    assert controller.find_run("Never asked?", tmp_path) is None
