"""Eval scoring (T12.1, T12.4, T12.5, T12.7): each PRD §8 metric on synthetic saved runs."""

import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from causal_genie.agent.state import (
    DagEdge,
    DatasetCandidate,
    Download,
    Query,
    QueryGroup,
    ReportDraft,
    ResearchPlan,
    ResearchRound,
    Result,
    RunState,
    Sentence,
    SourceKind,
)

EVAL_DIR = Path(__file__).parents[1] / "eval"
spec = importlib.util.spec_from_file_location("run_eval", EVAL_DIR / "run_eval.py")
assert spec is not None and spec.loader is not None
run_eval = importlib.util.module_from_spec(spec)
sys.modules["run_eval"] = run_eval
spec.loader.exec_module(run_eval)

CASES = {c["id"]: c for c in yaml.safe_load((EVAL_DIR / "cases.yaml").read_text())["cases"]}
RD = "https://vincentarelbundock.github.io/Rdatasets/csv"


def accepted(url: str, rank: int = 1, source: SourceKind = "repository") -> DatasetCandidate:
    return DatasetCandidate(
        url=url,
        host="x",
        license=None,
        access="open",
        coverage={},
        score=1.0,
        rejected_reason=None,
        source=source,
        title="t",
        description="",
        rank=rank,
        accepted=True,
    )


def prose(*texts: str, cites: tuple[str, ...] = ("c1",)) -> ReportDraft:
    return ReportDraft(
        summary=[Sentence(text=t, cites=list(cites)) for t in texts], comparison=[], limitations=[]
    )


# --- verdict, key studies, grounding, disproving search ---


def test_verdict_match(sample_state: RunState) -> None:
    sample_state.verdict = "estimated"
    assert run_eval.verdict_match(sample_state, CASES["D1"])
    sample_state.verdict = "published_only"
    assert not run_eval.verdict_match(sample_state, CASES["D1"])


def test_key_study_found_only_if_included_matching_pmid_doi_or_author_year(
    sample_state: RunState,
) -> None:
    keys = [
        {"pmid": "23921656", "doi": None, "first_author": "X", "year": 1},
        {"pmid": None, "doi": "10.1213/ane.0B013E318299A650", "first_author": "X", "year": 1},
        {"pmid": None, "doi": None, "first_author": "Ruetzler", "year": 2013},
        {"pmid": "1", "doi": "10.1/x", "first_author": "Nobody", "year": 1999},
    ]
    found, missing = run_eval.key_studies(sample_state, keys)
    assert (found, [m["first_author"] for m in missing]) == (3, ["Nobody"])

    sample_state.studies[0].included = False
    assert run_eval.key_studies(sample_state, keys)[0] == 0


def test_grounding_counts_quoted_and_checked_claims(sample_state: RunState) -> None:
    c = sample_state.claims[0]
    sample_state.claims = [
        c,
        c.model_copy(update={"id": "c2", "grounding": "quote_not_found", "support": "unchecked"}),
        c.model_copy(update={"id": "c3", "support": "partial"}),
        c.model_copy(update={"id": "c4", "grounding": "n/a", "support": "contradicts"}),
    ]
    g = run_eval.grounding(sample_state)
    assert (g["quoted"], g["quote_found"]) == (3, 2)
    assert (g["checked"], g["supports"], g["partial"]) == (3, 1, 1)


def test_disproving_search_ran_if_any_round_has_an_evidence_against_query(
    sample_state: RunState,
) -> None:
    def round_with(group: QueryGroup) -> ResearchRound:
        q = Query(text="q", tool="search", group=group, why="w")
        plan = ResearchPlan(queries=[q], done=False, done_reason="")
        return ResearchRound(number=1, plan=plan, planned_by="model")

    sample_state.rounds = [round_with("best_answer")]
    assert not run_eval.disproving_ran(sample_state)
    sample_state.rounds.append(round_with("evidence_against"))
    assert run_eval.disproving_ran(sample_state)


# --- report checks: unsupported numbers, forbidden conclusions, Genie, retractions ---


def test_unsupported_numbers_rerun_the_report_lint(sample_state: RunState) -> None:
    sample_state.prose = prose("Ruetzler 2013 found less sore throat.", "It fell by 99%.")
    assert run_eval.unsupported_numbers(sample_state) == ["It fell by 99%."]
    sample_state.prose = None
    assert run_eval.unsupported_numbers(sample_state) == []


def test_forbidden_conclusions_counts_judged_sentences_only(make_llm, sample_state) -> None:
    sample_state.prose = prose(
        "There is no good evidence that licorice has no effect.", "Licorice has no effect."
    )
    judged = {
        "conclusions": [
            {"conclusion": 1, "asserted": False, "sentence": None},
            {"conclusion": 2, "asserted": True, "sentence": 2},
        ]
    }
    llm, fake = make_llm([json.dumps(judged)])

    hits = run_eval.forbidden_conclusions(sample_state, CASES["D1"], llm)

    assert hits == [
        {
            "conclusion": CASES["D1"]["forbidden_conclusions"][1],
            "sentence": "Licorice has no effect.",
        }
    ]
    (request,) = fake.requests
    user = request["messages"][1]["content"]
    assert "[2] Licorice has no effect." in user and "There is no good evidence" in user
    # the small tier flipped on live D1's "likely reduces" (1 of 2 tries); large, reasoning off,
    # flagged nothing wrong in 8 tries on the 4 scored reports
    assert request["model"] == "test/large"
    assert request["extra_body"] == {"chat_template_kwargs": {"thinking": False}}


def test_forbidden_judge_pointing_at_no_real_sentence_is_an_error(make_llm, sample_state) -> None:
    sample_state.prose = prose("Licorice helps.")
    judged = {"conclusions": [{"conclusion": 1, "asserted": True, "sentence": 7}]}
    llm, _ = make_llm([json.dumps(judged)])

    with pytest.raises(ValueError, match="sentence 7"):
        run_eval.forbidden_conclusions(sample_state, CASES["D1"], llm)


def test_d4_forbids_any_estimate_by_rule_without_a_model(make_llm, sample_state) -> None:
    llm, fake = make_llm([])
    assert run_eval.forbidden_conclusions(sample_state, CASES["D4"], llm) == []
    sample_state.results = [
        Result(
            id="res-1",
            tool="rct_effect",
            tool_version="1",
            inputs_hash="x",
            outputs={"estimate": 0.05},
        )
    ]
    hits = run_eval.forbidden_conclusions(sample_state, CASES["D4"], llm)
    assert [h["conclusion"] for h in hits] == ["Any numeric effect estimate."]
    assert fake.requests == []


def test_genie_shown_where_not_allowed_fails(sample_state: RunState) -> None:
    sample_state.genie = "A playful aside."
    assert not run_eval.genie_ok(sample_state, CASES["G1"])
    assert run_eval.genie_ok(sample_state, CASES["D5"])
    sample_state.genie = None
    assert run_eval.genie_ok(sample_state, CASES["G1"])


def test_retracted_key_papers_must_be_flagged_if_found(sample_state: RunState) -> None:
    s = sample_state.studies[0]
    sample_state.studies = [s.model_copy(update={"pmid": "9500320", "retracted": False})]
    assert run_eval.unflagged_retractions(sample_state, CASES["G1"]) == ["Wakefield 1998"]
    sample_state.studies[0].retracted = True
    assert run_eval.unflagged_retractions(sample_state, CASES["G1"]) == []
    assert run_eval.unflagged_retractions(sample_state, CASES["D1"]) == []


# --- data check (T12.7) and DAG check (T12.5) ---


@pytest.mark.parametrize(
    "url",
    [
        f"{RD}/fivethirtyeight/US_births_1994_2003.csv",
        f"{RD}/fivethirtyeight/US_births_2000_2014.csv",
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_2000-2014_SSA.csv",
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_1994-2003_CDC_NCHS.csv",
        f"{RD}/mosaicData/Birthdays.csv",  # daily births by US state, 1969-1988 (user OK)
    ],
)
def test_d0_accepts_a_daily_us_births_table(sample_state: RunState, url: str) -> None:
    sample_state.datasets = [accepted(url, rank=2)]
    assert run_eval.data_check(sample_state, CASES["D0"]) == {"ok": True, "rank": 2, "url": url}


def test_data_check_other_cases(sample_state: RunState) -> None:
    sample_state.datasets = [accepted(f"{RD}/mosaicData/Births.csv")]
    assert run_eval.data_check(sample_state, CASES["D0"])["ok"] is False
    assert run_eval.data_check(sample_state, CASES["D5"])["ok"] is False  # expects no table
    sample_state.datasets = []
    assert run_eval.data_check(sample_state, CASES["D5"])["ok"] is True
    assert run_eval.data_check(sample_state, CASES["D1"])["ok"] is False
    sample_state.datasets = [accepted("tests/fixtures/d3_onboarding_ab.csv", source="user")]
    assert run_eval.data_check(sample_state, CASES["D3"])["ok"] is None  # user's table: n/a


def test_dag_check_d0(sample_state: RunState) -> None:
    assert sample_state.query is not None
    sample_state.query.exposure.name, sample_state.query.outcome.name = "full_moon", "births"
    edge = DagEdge.model_construct  # no validation needed for test edges
    sample_state.edges = [
        edge(source="full_moon", target="births", claim_ids=[]),
        edge(source="day_of_week", target="births", claim_ids=[]),
    ]
    assert run_eval.dag_problems(sample_state, CASES["D0"]) == []

    sample_state.edges += [
        edge(source="season", target="full_moon", claim_ids=[]),
        edge(source="full_moon", target="daytime", claim_ids=[]),
        edge(source="births", target="staffing", claim_ids=[]),
    ]
    sample_state.edges = [e for e in sample_state.edges if e.source != "day_of_week"]
    assert run_eval.dag_problems(sample_state, CASES["D0"]) == [
        "missing required node day_of_week",
        "edge into the exposure: season -> full_moon",
        "forbidden edge: full_moon -> daytime",
        "edge out of the outcome: births -> staffing",
    ]


def test_dag_check_required_node_and_no_dag(sample_state: RunState) -> None:
    assert run_eval.dag_problems(sample_state, CASES["D5"]) == ["missing required node age"]
    assert run_eval.dag_problems(sample_state, CASES["G1"]) == []
    sample_state.edges = []
    assert run_eval.dag_problems(sample_state, CASES["G1"]) == ["no DAG"]


# --- picking runs, cost, scorecard ---


def saved(runs: Path, run_id: str, question: str, data: str | None = None) -> Path:
    state = RunState(run_id=run_id, question=question)
    if data:
        state.datasets = [accepted(data, source="user")]
        state.downloads = [
            Download(
                url=data, path=data, sha256="x", bytes=1, retrieved_at="t", doi=None, license=None
            )
        ]
    return state.save(runs)


def test_pick_run_takes_the_latest_run_of_the_case(tmp_path: Path) -> None:
    q = CASES["D3"]["question"]
    saved(tmp_path, "20261114-100000-aaaaaa", q, "tests/fixtures/d3_onboarding_ab.csv")
    d4 = saved(tmp_path, "20261114-110000-bbbbbb", q, "tests/fixtures/d4_onboarding_completers.csv")
    d3 = saved(tmp_path, "20261114-120000-cccccc", q, "tests/fixtures/d3_onboarding_ab.csv")
    saved(tmp_path, "20261114-130000-dddddd", q)  # same question, searched data: neither case

    assert run_eval.pick_run(CASES["D3"], tmp_path) == d3
    assert run_eval.pick_run(CASES["D4"], tmp_path) == d4
    assert run_eval.pick_run(CASES["G1"], tmp_path) is None
    d5 = saved(tmp_path, "20261114-140000-eeeeee", CASES["D5"]["question"])
    assert run_eval.pick_run(CASES["D5"], tmp_path) == d5


def test_cost_sums_calls_and_wall_time_from_run_start_to_save(
    sample_state: RunState, tmp_path: Path
) -> None:
    sample_state.run_id = "20261114-220000-abcdef"
    path = sample_state.save(tmp_path)
    end = datetime(2026, 11, 14, 22, 0, tzinfo=UTC).timestamp() + 6400
    os.utime(path, (end, end))
    cost = run_eval.cost(sample_state, path)
    assert cost == {"tokens_in": 10, "tokens_out": 5, "calls": 1, "llm_s": 0.1, "wall_s": 6400}


def test_scorecard_counts_live_and_replay_separately(tmp_path: Path) -> None:
    rows = [
        {
            "case": "D0",
            "run_id": "r0",
            "expected": "estimated",
            "verdict": "estimated",
            "verdict_ok": True,
            "replay_verdict": "estimated",
            "replay_ok": True,
            "replay_error": None,
        },
        {
            "case": "D1",
            "run_id": "r1",
            "expected": "estimated",
            "verdict": "estimated",
            "verdict_ok": True,
            "replay_verdict": None,
            "replay_ok": False,
            "replay_error": "KeyError: no recorded call",
        },
        {
            "case": "G1",
            "run_id": None,
            "expected": "published_only",
            "verdict": None,
            "verdict_ok": False,
            "replay_verdict": None,
            "replay_ok": False,
            "replay_error": "no saved run",
        },
    ]
    md = run_eval.scorecard_counts(rows)
    assert md == {"live": "2/3", "replay": "1/3"}


def test_scorecard_details_stay_on_one_line_per_case() -> None:
    row = {
        "case": "G1",
        "run_id": "r",
        "expected": "published_only",
        "verdict": None,
        "verdict_ok": False,
        "replay_verdict": None,
        "replay_ok": True,
        "replay_error": None,
        "failure": "ModelOutputError: bad\n  Invalid JSON\n    more",
        "key_studies": "0/0",
        "key_studies_missing": [],
        "grounding": {"quoted": 0, "quote_found": 0, "checked": 0, "supports": 0, "partial": 0},
        "unsupported_numbers": [],
        "forbidden": [],
        "disproving_search": True,
        "genie_ok": True,
        "unflagged_retractions": [],
        "data": {"ok": True, "rank": None, "url": None},
        "dag_problems": [],
        "cost": {"tokens_in": 1, "tokens_out": 1, "calls": 1, "llm_s": 0.1, "wall_s": 1},
    }
    details = run_eval.scorecard_md([row]).split("## Details\n\n")[1]
    assert details.count("\n") == 1 and "Invalid JSON" in details
