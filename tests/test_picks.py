"""The nightly pick (U15, U16): which board question the Genie answers."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine

from causal_genie.agent.state import RunState
from causal_genie.ui import board_db, picks
from causal_genie.ui.board_db import Question
from causal_genie.ui.picks import rank

T0 = datetime(2026, 11, 15, 12, 0)


def q(qid: int, votes: int, minutes: int = 0, status: str = "shown") -> Question:
    return Question(
        id=qid,
        text=f"Question {qid}?",
        normalized=f"question {qid}",
        created_at=T0 + timedelta(minutes=minutes),
        status=status,  # ty: ignore[invalid-argument-type]
        reason=None,
        run_id=None,
        announcement=None,
        answered_at=None,
        voter_id="v",
        votes=votes,
    )


def test_most_votes_first() -> None:
    assert [x.id for x in rank([q(1, 2), q(2, 5), q(3, 0)])] == [2, 1, 3]


def test_ties_go_to_the_oldest() -> None:
    assert [x.id for x in rank([q(1, 3, minutes=10), q(2, 3, minutes=0)])] == [2, 1]


def test_only_shown_questions_are_ranked() -> None:
    ranked = rank([q(1, 9, status="hidden"), q(2, 1), q(3, 9, status="answered")])
    assert [x.id for x in ranked] == [2]


def test_empty_board_ranks_nothing() -> None:
    assert rank([]) == []


# --- U16: the nightly job -------------------------------------------------------------------


@pytest.fixture
def db() -> Engine:
    return board_db.connect("sqlite://")


def _shown(db: Engine, text: str, votes: int = 0, age: timedelta = timedelta(days=2)) -> int:
    qid = board_db.add_question(db, text, f"asker of {text}")
    board_db.set_status(db, qid, "shown")
    with db.begin() as conn:
        conn.execute(
            board_db.questions.update()
            .where(board_db.questions.c.id == qid)
            .values(created_at=board_db.now() - age)
        )
    for i in range(votes):
        board_db.vote(db, qid, f"voter {i}")
    return qid


class FakeRuns:
    """Stands in for a live run: saves a finished, partial or non-causal run per question."""

    def __init__(self, outcomes: dict[str, str]) -> None:
        self.outcomes = outcomes  # question -> "ok" | "partial" | "not_causal"
        self.asked: list[str] = []

    def __call__(self, question: str, runs_dir: Path) -> RunState:
        self.asked.append(question)
        outcome = self.outcomes.get(question, "ok")
        state = RunState(run_id=f"run-{len(self.asked)}", question=question)
        state.verdict = "not_causal" if outcome == "not_causal" else "published_only"
        if outcome == "partial":
            state.verdict, state.failure = None, "APIError: provider down"
        state.save(runs_dir)
        return state


def _nightly(db, tmp_path, runs: FakeRuns, n: int = 1) -> picks.Nightly:
    return picks.nightly(db, n, runs, tmp_path / "runs", tmp_path / "gallery")


def test_one_pick_is_answered_exported_and_described(db, tmp_path) -> None:
    top = _shown(db, "Does coffee cause headaches?", votes=3)
    _shown(db, "Does tea cause calm?", votes=1)
    runs = FakeRuns({})

    night = _nightly(db, tmp_path, runs)

    assert runs.asked == ["Does coffee cause headaches?"]
    (pick,) = night.picks
    assert (pick.question_id, pick.run_id, pick.votes) == (top, "run-1", 3)
    saved = json.loads((tmp_path / "gallery" / "run-1" / "pick.json").read_text())
    assert saved["question_id"] == top and saved["votes"] == 3 and saved["announcement"]
    assert (tmp_path / "gallery" / "run-1" / "state.json").exists()
    q = board_db.get_question(db, top)
    assert (q.status, q.run_id) == ("running", "run-1")
    assert not night.failed


def test_questions_younger_than_a_day_wait(db, tmp_path) -> None:
    _shown(db, "Does ashwagandha ease stress?", votes=5, age=timedelta(hours=3))
    old = _shown(db, "Does tea cause calm?", votes=0, age=timedelta(hours=25))

    night = _nightly(db, tmp_path, FakeRuns({}))

    assert [p.question_id for p in night.picks] == [old]


def test_no_pick_when_every_question_is_fresh(db, tmp_path) -> None:
    fresh = _shown(db, "Does ashwagandha ease stress?", votes=5, age=timedelta(hours=23))

    night = _nightly(db, tmp_path, FakeRuns({}))

    assert night.picks == [] and not night.failed
    assert board_db.get_question(db, fresh).status == "shown"


def test_a_failed_pick_falls_back_to_the_next(db, tmp_path) -> None:
    first = _shown(db, "Does coffee cause headaches?", votes=3)
    second = _shown(db, "Does tea cause calm?", votes=1)

    night = _nightly(db, tmp_path, FakeRuns({"Does coffee cause headaches?": "partial"}))

    assert [p.question_id for p in night.picks] == [second]
    skipped = board_db.get_question(db, first)
    assert skipped.status == "skipped" and "partial run" in (skipped.reason or "")
    assert not (tmp_path / "gallery" / "run-1").exists()


def test_two_failures_leave_the_slot_empty_and_fail_the_job(db, tmp_path) -> None:
    a = _shown(db, "Does coffee cause headaches?", votes=3)
    b = _shown(db, "Is water wet?", votes=2)
    c = _shown(db, "Does tea cause calm?", votes=1)
    runs = FakeRuns({"Does coffee cause headaches?": "partial", "Is water wet?": "not_causal"})

    night = _nightly(db, tmp_path, runs)

    assert night.picks == [] and night.failed
    assert len(runs.asked) == 2
    assert board_db.get_question(db, b).reason == "not a cause-and-effect question"
    assert [board_db.get_question(db, q).status for q in (a, b, c)] == [
        "skipped",
        "skipped",
        "shown",
    ]


def test_n_answers_up_to_n_questions(db, tmp_path) -> None:
    for i, text in enumerate(["Does A cause B?", "Does C cause D?", "Does E cause F?"]):
        _shown(db, text, votes=3 - i)

    night = _nightly(db, tmp_path, FakeRuns({}), n=2)

    assert [p.question for p in night.picks] == ["Does A cause B?", "Does C cause D?"]


def test_no_candidates_runs_nothing(db, tmp_path) -> None:
    runs = FakeRuns({})

    night = _nightly(db, tmp_path, runs)

    assert runs.asked == [] and night.picks == [] and not night.failed


def test_a_crashed_run_puts_the_question_back_and_raises(db, tmp_path) -> None:
    qid = _shown(db, "Does coffee cause headaches?")

    def crash(question: str, runs_dir: Path) -> RunState:
        raise KeyError("TAVILY_API_KEY")

    with pytest.raises(KeyError):
        picks.nightly(db, 1, crash, tmp_path / "runs", tmp_path / "gallery")
    assert board_db.get_question(db, qid).status == "shown"


def test_pr_body_keeps_stranger_text_inert(db, tmp_path) -> None:
    text = "Does @octocat ``` <b>x</b> [link](https://evil.example) cause harm?"
    _shown(db, text, votes=2)
    night = _nightly(db, tmp_path, FakeRuns({}))

    body = picks.pr_body(night)

    block = body.split("```text\n", 1)[1].split("\n```", 1)[0]
    assert block == text.replace("`", "'")
    assert body.count("```") == 4  # the question and the Genie's line, two fences each
    assert "run-1" in body and "2 votes" in body


def test_main_writes_the_pr_body_and_fails_when_a_slot_is_empty(db, tmp_path, monkeypatch) -> None:
    url = f"sqlite:///{tmp_path / 'board.db'}"
    board = board_db.connect(url)
    _shown(board, "Does coffee cause headaches?")
    _shown(board, "Is water wet?")
    monkeypatch.setenv("BOARD_DB_URL", url)
    monkeypatch.setattr(picks, "run_live", FakeRuns({}))
    args = ["run", "--n", "1", "--runs", str(tmp_path / "runs"), "--gallery"]
    args += [str(tmp_path / "g"), "--body-file", str(tmp_path / "body.md")]

    picks.main(args)
    assert "Does coffee cause headaches?" in (tmp_path / "body.md").read_text()

    (tmp_path / "body.md").unlink()
    monkeypatch.setattr(picks, "run_live", FakeRuns({"Is water wet?": "not_causal"}))
    with pytest.raises(SystemExit) as exit_:
        picks.main(args)
    assert exit_.value.code == 1 and not (tmp_path / "body.md").exists()


@pytest.mark.parametrize("n", ["0", "4", "two"])
def test_main_rejects_n_outside_1_to_3(n, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BOARD_DB_URL", "sqlite://")
    with pytest.raises(SystemExit):
        picks.main(["run", "--n", n, "--body-file", str(tmp_path / "b.md")])


WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"


@pytest.mark.parametrize("name", ["daily-genie.yml", "genie-publish.yml"])
def test_workflow_run_lines_never_expand_expressions(name) -> None:
    """Values reach the shell only through `env:`, never `${{ }}` pasted into a `run:` line."""
    import yaml

    workflow = yaml.safe_load((WORKFLOWS / name).read_text())
    steps = [s for job in workflow["jobs"].values() for s in job["steps"]]
    runs = [s["run"] for s in steps if "run" in s]
    assert runs and all("${{" not in r for r in runs)
    assert any("python -m causal_genie.ui." in r for r in runs)


# --- U17: the Genie's pick announcement -------------------------------------------------------


def pick_line(sensitive: bool = False, line: str = "The kettle is on and the jury is out.") -> str:
    """genie_pick model output as JSON."""
    return json.dumps({"sensitive": sensitive, "line": line})


def test_genie_line_on_a_light_topic(make_llm) -> None:
    llm, fake = make_llm([pick_line()])

    line = picks.genie_line("Does tea cause calm?", llm)

    assert line == "The kettle is on and the jury is out."
    assert [(c.prompt_name, c.tier) for c in llm.calls] == [("genie_pick", "large")]
    assert fake.requests[0]["messages"][-1]["content"] == (
        "<question>\nDoes tea cause calm?\n</question>"
    )


def test_no_genie_line_on_a_sensitive_topic(make_llm) -> None:
    llm, _ = make_llm([pick_line(sensitive=True)])

    assert picks.genie_line("Do vaccines cause autism?", llm) is None


def test_no_genie_line_with_a_number(make_llm) -> None:
    llm, _ = make_llm([pick_line(line="Over 9000 votes can't be wrong!")])

    assert picks.genie_line("Does tea cause calm?", llm) is None


def test_a_number_from_the_question_is_allowed(make_llm) -> None:
    llm, _ = make_llm([pick_line(line="Four days a week, all of them curious.")])
    assert picks.genie_line("Does a 4-day week raise output?", llm) is not None
    llm, _ = make_llm([pick_line(line="A 4-day week walks into a lab.")])
    assert picks.genie_line("Does a 4-day week raise output?", llm) is not None


def test_the_pick_uses_the_plain_line_without_the_genie(db, tmp_path) -> None:
    _shown(db, "Do vaccines cause autism?")

    night = picks.nightly(
        db, 1, FakeRuns({}), tmp_path / "runs", tmp_path / "g", announce=lambda q: None
    )

    (pick,) = night.picks
    assert pick.announcement == "Tonight's question: Do vaccines cause autism?"
    assert pick.genie is False


def test_the_pick_keeps_the_genie_line(db, tmp_path) -> None:
    _shown(db, "Does tea cause calm?")

    night = picks.nightly(
        db, 1, FakeRuns({}), tmp_path / "runs", tmp_path / "g", announce=lambda q: "Brew!"
    )

    assert (night.picks[0].announcement, night.picks[0].genie) == ("Brew!", True)


def test_genie_pick_runs_without_reasoning() -> None:
    from causal_genie.llm.client import load_no_thinking

    assert "genie_pick" in load_no_thinking()
