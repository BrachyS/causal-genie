import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import httpx
import httpx2
import openai
import pytest
from conftest import FIXTURES, FakeTavily, epmc_result, extraction, proposal
from tavily import TavilyClient

from causal_genie.agent import controller
from causal_genie.agent import state as state_module
from causal_genie.agent.state import RunState
from causal_genie.agent.steps.data import RDATASETS, RDATASETS_INDEX
from causal_genie.llm.client import LLMClient
from causal_genie.tools import download as dl
from causal_genie.tools.download import Fetcher
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.search import Searcher

D1 = "Does licorice gargling reduce post-op sore throat?"
KNUCKLES = "Does cracking your knuckles cause arthritis?"
PUBMED = "https://pubmed.ncbi.nlm.nih.gov/23921656/"
STUDY = "s" + hashlib.sha256(PUBMED.encode()).hexdigest()[:8]  # the study id research gives it
LICORICE = f"{RDATASETS}/csv/medicaldata/licorice_gargle.csv"
LICORICE_DOCS = "https://higgi13425.github.io/medicaldata/reference/licorice_gargle.html"
INDEX_HEADER = '"Package","Item","Title","Rows","Cols","CSV","Doc"'
INDEX = (
    f'{INDEX_HEADER}\n"medicaldata","licorice_gargle","Licorice Gargle before Intubation",'
    f'8,3,"{LICORICE}","{RDATASETS}/doc/medicaldata/licorice_gargle.html"\n'
).encode()
# 3/4 vs 1/3 with sore throat (pain > 0); one row has no outcome
CSV = (
    b"rownames,treat,pacu30min_throatPain\n1,0,3\n2,0,1\n3,0,0\n4,0,2\n5,1,0\n6,1,4\n7,1,0\n8,1,\n"
)
PAPER = "A randomized, double-blind trial. Sore throat at 30 minutes was 24% with licorice."
DESIGN_QUOTE = [{"field": "design", "text": "A randomized, double-blind trial"}]

READ = extraction(quotes=DESIGN_QUOTE, data_link=LICORICE_DOCS)  # the D1 trial, with data link
SUPPORTED = json.dumps({"judgments": [{"claim_id": f"{STUDY}-design", "support": "supports"}]})
DAG = proposal(("licorice_gargle", "sore_throat", []))  # only the question edge: no extra round
COVERED = json.dumps(
    {
        "candidates": [
            {
                "candidate": 1,
                "coverage": [
                    {"node": "licorice_gargle", "column": "treat"},
                    {"node": "sore_throat", "column": "pacu30min_throatPain"},
                ],
            }
        ]
    }
)


def mapping(exposure: dict, outcome: dict) -> str:
    """Map-columns model output for the D1 nodes."""
    return json.dumps(
        {
            "nodes": [
                {"node": "licorice_gargle", "derive": None, "cutoff": None, "categorical": False}
                | exposure,
                {"node": "sore_throat", "derive": None, "cutoff": None, "categorical": False}
                | outcome,
            ],
            "unit": None,
            "time": None,
        }
    )


MAPPED = mapping(
    {"columns": ["treat"]},
    {"columns": ["pacu30min_throatPain"], "derive": "cutoff", "cutoff": 0},
)
# report model output: one summary sentence citing the verdict
PROSE = json.dumps(
    {
        "summary": [{"text": "The controller reached a verdict.", "cites": ["F-verdict"]}],
        "comparison": [],
        "limitations": [],
    }
)
D1_CALLS = [READ, SUPPORTED, DAG, COVERED, MAPPED, PROSE]  # after route, frame and the round-1 plan


def routed(causal: bool) -> str:
    """Router model output saying whether the question is causal."""
    return json.dumps({"causal": causal, "reason": "test", "causal_rewrite": None})


@pytest.fixture
def framed(sample_state: RunState) -> str:
    """Frame model output: the sample run's estimand as JSON."""
    assert sample_state.query is not None
    return sample_state.query.model_dump_json()


class TrialTavily(FakeTavily):
    """Fake Tavily whose one hit carries the trial's text, so quotes from it are grounded."""

    def search(self, query: str, **kwargs) -> dict:
        found = super().search(query, **kwargs)
        found["results"][0]["content"] = PAPER
        return found


@pytest.fixture
def served() -> dict[str, bytes]:
    """URL → body served by the fake web; tests change entries."""
    return {RDATASETS_INDEX: INDEX, LICORICE: CSV}


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch, served: dict[str, bytes]) -> httpx.Client:
    """HTTP client serving `served` (404 otherwise), with DNS faked to a public IP."""
    monkeypatch.setattr(dl, "resolve_ips", lambda host: ["185.199.108.153"])

    def handle(request: httpx.Request) -> httpx.Response:
        body = served.get(str(request.url))
        return httpx.Response(404) if body is None else httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handle))


@pytest.fixture
def tavily() -> TrialTavily:
    return TrialTavily()


@pytest.fixture
def run_live(make_llm, make_epmc, tavily, http, tmp_path: Path):
    """Run the controller end to end on fakes, given the model outputs in order."""

    def run(question: str, outcomes: list, ask=None, **options) -> RunState:
        llm, _ = make_llm(outcomes)
        searcher = Searcher(cast(TavilyClient, tavily))
        papers, _ = make_epmc([epmc_result()])
        return controller.run(
            question,
            llm,
            searcher,
            papers,
            Fetcher(http),
            tmp_path / "runs",
            tmp_path / "cache",
            ask=ask,
            **options,
        )

    return run


def test_d1_runs_end_to_end_and_saves_the_run(run_live, framed, round_plan, tmp_path: Path) -> None:
    state = run_live(D1, [routed(True), framed, round_plan, *D1_CALLS])

    assert state.verdict == "estimated"
    assert state.design is not None and state.design.design == "randomized"
    (result,) = state.results
    assert result.tool == "rct_effect"
    out = result.outputs
    assert (out["events_treated"], out["n_treated"]) == (1, 3)
    assert (out["events_control"], out["n_control"]) == (3, 4)
    assert out["rows_excluded_missing"] == 1
    assert out["mapped_from"] == {
        "licorice_gargle": "treat",
        "sore_throat": "pacu30min_throatPain > 0",
    }
    accepted = next(c for c in state.datasets if c.accepted)
    assert (accepted.url, accepted.source, accepted.study_id) == (LICORICE, "paper", STUDY)
    assert state.mapping_confirmed is False  # nobody to ask
    assert [r.planned_by for r in state.rounds] == ["model"]
    assert [(e.source, e.target, e.question, e.assumed) for e in state.edges] == [
        ("licorice_gargle", "sore_throat", True, False)
    ]
    assert state.downloads[-1].license == "MIT"
    assert [c.prompt_name for c in state.llm_calls] == [
        "route",
        "frame",
        "research_plan",
        "extract",
        "check_support",
        "dag",
        "data_coverage",
        "map",
        "report",
    ]
    assert state.report is not None and "✅" in state.report
    assert "## Evidence" in state.report and "Risk difference" in state.report
    assert "The controller reached a verdict." in state.report
    run_dir = tmp_path / "runs" / state.run_id
    assert RunState.load(run_dir / "state.json") == state
    assert (run_dir / "report.md").read_text() == state.report
    (saved,) = json.loads((run_dir / "dag.json").read_text())["edges"]
    assert saved["question"] is True and saved["assumed"] is False
    assert (run_dir / "dag.png").read_bytes().startswith(b"\x89PNG")


def test_trial_data_without_a_documented_design_is_analyzed_as_observational(
    run_live, framed, round_plan
) -> None:
    undocumented = extraction(quotes=[], data_link=LICORICE_DOCS)  # no design quote, no check

    state = run_live(
        D1, [routed(True), framed, round_plan, undocumented, DAG, COVERED, MAPPED, PROSE]
    )

    assert state.verdict == "estimated"
    assert state.design is not None and state.design.design == "observational"
    (result,) = state.results
    assert result.tool == "backdoor_effect"
    assert state.adjustment is not None and state.adjustment.adjust_for == []


def test_a_mapped_column_that_is_all_missing_is_left_out_of_the_estimate(
    run_live, framed, round_plan, served
) -> None:
    # Live D0 (run 20261115-221659-ff7ee1): `day_of_week` derived from a weekday-name column
    # ("Wed") gave no dates, so every row was missing and the estimate had no rows at all.
    served[LICORICE] = b"rownames,treat,pacu30min_throatPain,wday\n" + b"".join(
        row + b",Wed\n" for row in CSV.splitlines()[1:]
    )
    undocumented = extraction(quotes=[], data_link=LICORICE_DOCS)
    dag = proposal(("licorice_gargle", "sore_throat", []), ("weekday", "sore_throat", []))
    weekday = {"node": "weekday", "columns": ["wday"], "derive": "day_of_week"}
    mapped = json.loads(MAPPED)
    mapped["nodes"].append({"cutoff": None, "categorical": True} | weekday)

    state = run_live(
        D1,
        [routed(True), framed, round_plan, undocumented, dag, COVERED, json.dumps(mapped), PROSE],
    )

    assert state.verdict == "estimated"
    (result,) = state.results
    assert result.tool == "backdoor_effect"
    assert state.adjustment is not None and "weekday" not in state.adjustment.precision
    assert result.outputs["n"] == 7  # the one row without an outcome is the only one excluded


def test_user_can_say_assignment_was_random(run_live, framed, round_plan) -> None:
    undocumented = extraction(quotes=[], data_link=LICORICE_DOCS)
    asked: list[str] = []

    def ask(question: str) -> str:
        asked.append(question)
        return "yes"

    state = run_live(
        D1, [routed(True), framed, round_plan, undocumented, DAG, COVERED, MAPPED, PROSE], ask=ask
    )

    assert state.design is not None and state.design.design == "randomized"
    assert state.results[0].tool == "rct_effect"
    assert len(asked) == 2  # the mapping, then whether assignment was random
    assert "randomly assigned" in asked[1]
    assert state.clarifications == {q: "yes" for q in asked}  # recorded for replay


def test_user_is_not_asked_about_randomization_when_the_paper_documents_it(
    run_live, framed, round_plan
) -> None:
    asked: list[str] = []

    def ask(question: str) -> str:
        asked.append(question)
        return "yes"

    run_live(D1, [routed(True), framed, round_plan, *D1_CALLS], ask=ask)

    assert len(asked) == 1 and "mapping" in asked[0].lower()


def test_selection_after_assignment_gives_a_study_plan(
    run_live, framed, round_plan, served
) -> None:
    served[LICORICE] = (FIXTURES / "d4_onboarding_completers.csv").read_bytes()
    covered = COVERED.replace("pacu30min_throatPain", "activated_7d").replace("treat", "new_flow")
    d4_mapping = mapping({"columns": ["new_flow"]}, {"columns": ["activated_7d"]})

    state = run_live(
        D1, [routed(True), framed, round_plan, READ, SUPPORTED, DAG, covered, d4_mapping, PROSE]
    )

    assert state.verdict == "study_plan"
    assert state.design is not None and state.design.design == "selection_after_assignment"
    assert state.results == []
    assert state.report is not None and "## Study plan" in state.report


def test_restricted_data_only_gives_restricted(run_live, framed, round_plan, served) -> None:
    served[RDATASETS_INDEX] = INDEX_HEADER.encode() + b"\n"
    vivli = extraction(quotes=[], data_link="https://vivli.org/study/123.csv")

    state = run_live(D1, [routed(True), framed, round_plan, vivli, DAG, PROSE])

    assert state.verdict == "restricted"
    assert any(c.access == "restricted" for c in state.datasets)


def test_studies_but_no_usable_data_is_published_only(run_live, framed, round_plan, served) -> None:
    served[RDATASETS_INDEX] = INDEX_HEADER.encode() + b"\n"

    state = run_live(
        KNUCKLES, [routed(True), framed, round_plan, extraction(quotes=[]), DAG, PROSE]
    )

    assert state.verdict == "published_only"
    assert state.downloads[1:] == [] and state.results == []  # only the Rdatasets index
    assert state.datasets and all(c.rejected_reason for c in state.datasets)


def test_not_causal_question_stops_after_routing(run_live, tavily) -> None:
    state = run_live("How many babies are born each day?", [routed(False)])

    assert state.verdict == "not_causal"
    assert tavily.requests == []
    assert state.query is None


def test_no_included_study_and_no_data_is_insufficient(
    run_live, framed, round_plan, served
) -> None:
    served[RDATASETS_INDEX] = INDEX_HEADER.encode() + b"\n"
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

    state = run_live(KNUCKLES, [routed(True), framed, round_plan, off_topic, round_2, PROSE])

    assert state.verdict == "insufficient"
    assert state.edges == []  # no evidence to build a graph from
    assert state.report is not None and "No included studies" in state.report


@pytest.fixture
def fake_clock(monkeypatch):
    """Each reading of the clock is one second after the last."""
    ticks = (datetime(2026, 11, 15, tzinfo=UTC) + timedelta(seconds=s) for s in range(1000))
    monkeypatch.setattr(state_module, "now", lambda: next(ticks))


def test_each_step_and_research_round_is_timed_in_order(
    run_live, framed, round_plan, fake_clock, tmp_path: Path
) -> None:
    state = run_live(D1, [routed(True), framed, round_plan, *D1_CALLS])

    assert [t.name for t in state.timings] == [
        "route",
        "frame",
        "research",
        "research round 1",
        "dag",
        "data",
        "map",
        "estimate",
        "report",
    ]
    starts = [t.started_at for t in state.timings]
    assert starts == sorted(starts)
    assert all(t.seconds is not None and t.seconds > 0 for t in state.timings)
    research, round_1 = state.timings[2:4]
    assert round_1.seconds is not None and research.seconds is not None
    assert round_1.seconds < research.seconds  # the round runs inside the research step
    saved = RunState.load(tmp_path / "runs" / state.run_id / "state.json")
    assert saved.timings == state.timings


def test_a_failed_step_is_still_timed_and_named(run_live, framed, fake_clock) -> None:
    timeout = openai.APITimeoutError(request=httpx2.Request("POST", "https://api.test"))

    state = run_live(D1, [routed(True), framed, timeout, timeout])

    assert state.step == "research"  # a round's timing does not rename the failed step
    assert [t.name for t in state.timings] == ["route", "frame", "research", "research round 1"]
    assert all(t.seconds is not None for t in state.timings)


def test_failed_step_still_saves_the_run(run_live, tmp_path: Path) -> None:
    with pytest.raises(IndexError):  # the fake model has no answer for the frame call
        run_live(D1, [routed(True)])

    (saved,) = (tmp_path / "runs").glob("*/state.json")
    state = RunState.load(saved)
    assert state.verdict is None
    assert [c.prompt_name for c in state.llm_calls] == ["route"]


def test_replay_reproduces_a_run_offline(
    run_live, framed, round_plan, prompts_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    live = run_live(D1, [routed(True), framed, round_plan, *D1_CALLS])
    saved = RunState.load(tmp_path / "runs" / live.run_id / "state.json")
    assert [c.kind for c in saved.fetches] == ["download", "peek", "download"]

    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: pytest.fail("HTTP request"))
    monkeypatch.setattr(dl, "resolve_ips", lambda host: pytest.fail(f"DNS lookup of {host}"))
    replayed = controller.run(
        D1,
        LLMClient.from_replay(saved.llm_calls, prompts_dir=prompts_dir),
        Searcher.from_replay(saved.searches, saved.extracts),
        EuropePMC.from_replay(saved.lookups),
        Fetcher.from_replay(saved.fetches),
        tmp_path / "runs",
        tmp_path / "cache",
        ask=None,
    )

    assert replayed.results == live.results
    assert replayed.rounds == live.rounds
    assert replayed.edges == live.edges
    assert replayed.design == live.design
    assert replayed.datasets == live.datasets
    assert replayed.downloads == live.downloads  # read from cache/ by checksum
    assert replayed.fetches == live.fetches
    assert replayed.report is not None and live.report is not None
    assert replayed.report.replace(replayed.run_id, "") == live.report.replace(live.run_id, "")


def test_replay_answers_clarifying_questions_from_the_run(
    run_live, sample_state: RunState, round_plan, prompts_dir: Path, tmp_path: Path
) -> None:
    assert sample_state.query is not None
    which = "Sore throat at which time after surgery?"
    ambiguous = sample_state.query.model_copy(update={"ambiguities": [which]})
    outcomes = [routed(True), ambiguous.model_dump_json(), sample_state.query.model_dump_json()]
    live = run_live(
        D1,
        [*outcomes, round_plan, *D1_CALLS],
        ask=lambda q: "30 minutes" if q == which else "yes",  # yes to the mapping
    )

    assert live.clarifications[which] == "30 minutes"
    assert live.mapping_confirmed and live.verdict == "estimated"
    replayed = controller.run(
        D1,
        LLMClient.from_replay(live.llm_calls, prompts_dir=prompts_dir),
        Searcher.from_replay(live.searches, live.extracts),
        EuropePMC.from_replay(live.lookups),
        Fetcher.from_replay(live.fetches),
        tmp_path / "runs",
        tmp_path / "cache",
        ask=live.clarifications.__getitem__,
    )

    assert replayed.query == live.query and replayed.clarifications == live.clarifications
    assert replayed.results == live.results


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


def test_genie_mode_adds_a_gated_aside(run_live, framed, round_plan, served) -> None:
    served[RDATASETS_INDEX] = INDEX_HEADER.encode() + b"\n"
    gate = json.dumps(
        {
            "sensitive": False,
            "whimsical": True,
            "aside": "Knuckles pop, joints stay calm.",
            "next_step": "Next: read the studies above.",
        }
    )
    outcomes = [routed(True), framed, round_plan, extraction(quotes=[]), DAG, PROSE, gate]

    state = run_live(KNUCKLES, outcomes, genie=True)

    assert state.verdict == "published_only"
    assert [c.prompt_name for c in state.llm_calls][-2:] == ["report", "genie_gate"]
    assert state.report is not None and "🧞 Genie's aside" in state.report


def test_api_failure_saves_a_partial_report_naming_the_step(
    run_live, framed, tmp_path: Path
) -> None:
    timeout = openai.APITimeoutError(request=httpx2.Request("POST", "https://api.test"))

    state = run_live(D1, [routed(True), framed, timeout, timeout])  # failed after one retry

    assert state.verdict is None and state.step == "research"
    assert state.failure is not None and "APITimeoutError" in state.failure
    assert state.report is not None and "Partial report" in state.report
    assert "`research` step failed" in state.report
    assert "no evidence exists" in state.report
    run_dir = tmp_path / "runs" / state.run_id
    assert RunState.load(run_dir / "state.json").failure == state.failure
    assert (run_dir / "report.md").read_text() == state.report


def test_schema_failure_after_repair_saves_a_partial_report(run_live, tmp_path: Path) -> None:
    state = run_live(D1, [routed(True), "not json", '{"wrong": 1}'])  # frame fails twice

    assert state.verdict is None and state.step == "frame"
    assert state.failure is not None and "ModelOutputError" in state.failure
    assert state.report is not None and "`frame` step failed" in state.report
    assert (tmp_path / "runs" / state.run_id / "report.md").read_text() == state.report


def user_data_run(run_live, framed, round_plan, fixture: str) -> RunState:
    """Run D1's framing on a user's table (D3/D4 fixtures); the user says yes to both the
    mapping and random assignment. No study is included, so no DAG and no data search."""
    off_topic = extraction(included=False, quotes=[])
    round_2 = json.dumps(
        {
            "queries": [
                {"text": "onboarding", "tool": "search", "group": "primary_studies", "why": "w"}
            ],
            "done": True,
            "done_reason": "test",
        }
    )
    ab_mapping = mapping({"columns": ["new_flow"]}, {"columns": ["activated_7d"]})
    return run_live(
        D1,
        [routed(True), framed, round_plan, off_topic, round_2, ab_mapping, PROSE],
        ask=lambda q: "yes",
        data=FIXTURES / fixture,
    )


def test_user_data_randomized_ab_test_is_estimated(run_live, framed, round_plan) -> None:
    state = user_data_run(run_live, framed, round_plan, "d3_onboarding_ab.csv")

    assert state.verdict == "estimated"
    assert "data_coverage" not in [c.prompt_name for c in state.llm_calls]
    (accepted,) = state.datasets
    assert accepted.source == "user"
    (result,) = state.results
    assert result.tool == "rct_effect"
    assert result.outputs["estimate"] == pytest.approx(0.055, abs=1e-9)
    assert result.outputs["ci_low"] < 0.05 < result.outputs["ci_high"]  # true effect inside


def test_user_data_completers_only_gives_a_study_plan(run_live, framed, round_plan) -> None:
    state = user_data_run(run_live, framed, round_plan, "d4_onboarding_completers.csv")

    assert state.verdict == "study_plan"
    assert state.design is not None and state.design.design == "selection_after_assignment"
    assert state.results == []


def test_cli_replay_runs_offline_without_keys(
    run_live, framed, round_plan, prompts_dir: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    live = run_live(D1, [routed(True), framed, round_plan, *D1_CALLS])
    capsys.readouterr()
    monkeypatch.setattr(controller, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(controller, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(controller, "load_dotenv", lambda: None)
    monkeypatch.setattr(controller, "setup_logging", lambda: None)
    real_replay = LLMClient.from_replay
    monkeypatch.setattr(
        LLMClient, "from_replay", lambda calls: real_replay(calls, prompts_dir=prompts_dir)
    )
    for key in ("NEBIUS_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CG_REPLAY", "1")
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: pytest.fail("HTTP request"))
    monkeypatch.setattr(dl, "resolve_ips", lambda host: pytest.fail(f"DNS lookup of {host}"))

    controller.main([D1])

    (replayed,) = [p for p in (tmp_path / "runs").iterdir() if p.name != live.run_id]
    assert live.report is not None
    out = capsys.readouterr().out
    assert out.replace(replayed.name, "") == live.report.replace(live.run_id, "")
