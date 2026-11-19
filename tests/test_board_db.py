"""Question of the Day board storage (U10): questions, votes and their status, on SQLite here."""

from datetime import timedelta

import pytest
from sqlalchemy import Engine, inspect

from causal_genie.ui import board_db


@pytest.fixture
def db() -> Engine:
    return board_db.connect("sqlite://")


def _shown(db: Engine, text: str, voter: str | None = None) -> int:
    qid = board_db.add_question(db, text, voter or f"asker of {text}")
    board_db.set_status(db, qid, "shown")
    return qid


def test_create_tables_is_idempotent(db) -> None:
    board_db.create_tables(db)
    board_db.create_tables(db)

    assert {"questions", "votes"} <= set(inspect(db).get_table_names())


def test_a_new_question_is_pending_with_its_normalized_text(db) -> None:
    qid = board_db.add_question(db, "  Does coffee cause headaches?  ", "v1")

    q = board_db.get_question(db, qid)
    assert q.text == "Does coffee cause headaches?"
    assert q.normalized == "does coffee cause headaches"
    assert q.status == "pending" and q.voter_id == "v1" and q.votes == 0


@pytest.mark.parametrize("text", ["", "   ", "x" * 201])
def test_empty_or_over_200_chars_is_rejected(db, text) -> None:
    with pytest.raises(ValueError):
        board_db.add_question(db, text, "v1")


def test_one_vote_per_voter_per_question(db) -> None:
    qid = _shown(db, "Does sleep improve memory?")

    assert board_db.vote(db, qid, "v1") is True
    assert board_db.vote(db, qid, "v1") is False
    assert board_db.vote(db, qid, "v2") is True

    assert board_db.get_question(db, qid).votes == 2
    assert board_db.voted(db, "v1") == {qid}


def test_only_shown_questions_take_votes(db) -> None:
    pending = board_db.add_question(db, "Does salt raise blood pressure?", "v1")

    with pytest.raises(ValueError, match="not open for votes"):
        board_db.vote(db, pending, "v2")
    with pytest.raises(KeyError):
        board_db.vote(db, pending + 100, "v2")


def test_board_lists_only_shown_questions_with_votes(db) -> None:
    older = _shown(db, "Does exercise reduce anxiety?")
    newer = _shown(db, "Does sugar cause hyperactivity?")
    top = _shown(db, "Does smoking cause cancer?")
    board_db.add_question(db, "Pending one?", "v9")
    hidden = _shown(db, "Hidden one?")
    board_db.set_status(db, hidden, "hidden", reason="spam")
    for voter in ("a", "b"):
        board_db.vote(db, top, voter)

    rows = board_db.board(db)

    assert [(q.id, q.votes) for q in rows] == [(older, 0), (newer, 0), (top, 2)]


def test_set_status_records_reason_and_answer(db) -> None:
    qid = _shown(db, "Does reading cause myopia?")

    board_db.set_status(db, qid, "skipped", reason="partial run")
    assert board_db.get_question(db, qid).reason == "partial run"

    board_db.set_status(db, qid, "answered", run_id="r1", announcement="Hmm!")
    q = board_db.get_question(db, qid)
    assert (q.status, q.run_id, q.announcement) == ("answered", "r1", "Hmm!")
    assert q.answered_at is not None


def test_unknown_status_or_question_is_an_error(db) -> None:
    qid = _shown(db, "Does tea lower blood pressure?")

    with pytest.raises(ValueError):
        board_db.set_status(db, qid, "deleted")  # ty: ignore[invalid-argument-type]
    with pytest.raises(KeyError):
        board_db.set_status(db, qid + 100, "shown")


def test_latest_answered_is_the_most_recent(db) -> None:
    assert board_db.latest_answered(db) is None
    first = _shown(db, "Does fish oil help the heart?")
    second = _shown(db, "Does meditation lower stress?")
    board_db.set_status(db, first, "answered", run_id="r1")
    board_db.set_status(db, second, "answered", run_id="r2")

    latest = board_db.latest_answered(db)

    assert latest is not None and latest.id == second
    assert [q.id for q in board_db.answered(db)] == [second, first]


def test_three_questions_a_day_per_voter(db, monkeypatch) -> None:
    start = board_db.now()
    monkeypatch.setattr(board_db, "now", lambda: start)
    for i in range(3):
        board_db.add_question(db, f"Does habit {i} cause sleep loss?", "v1")

    with pytest.raises(board_db.LimitReached, match="3 questions a day"):
        board_db.add_question(db, "Does habit 4 cause sleep loss?", "v1")
    board_db.add_question(db, "Another voter's question?", "v2")

    monkeypatch.setattr(board_db, "now", lambda: start + timedelta(hours=24, seconds=1))
    board_db.add_question(db, "Does habit 4 cause sleep loss?", "v1")


def test_thirty_votes_an_hour_per_voter(db, monkeypatch) -> None:
    start = board_db.now()
    monkeypatch.setattr(board_db, "now", lambda: start)
    ids = [_shown(db, f"Question {i}?", voter=f"asker{i}") for i in range(31)]
    for qid in ids[:30]:
        board_db.vote(db, qid, "v1")

    with pytest.raises(board_db.LimitReached, match="30 votes an hour"):
        board_db.vote(db, ids[30], "v1")
    assert board_db.get_question(db, ids[30]).votes == 0

    monkeypatch.setattr(board_db, "now", lambda: start + timedelta(hours=1, seconds=1))
    assert board_db.vote(db, ids[30], "v1") is True


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgres://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_postgres_urls_use_the_installed_driver(url, expected) -> None:
    assert board_db.driver_url(url) == expected


def test_the_engine_replaces_connections_the_server_dropped(tmp_path) -> None:
    # A hosted Postgres closes idle connections (e.g. during a 10-minute live run); without a
    # pre-ping the next board query fails and the app shows "Board unavailable".
    engine = board_db.connect(f"sqlite:///{tmp_path / 'board.db'}")

    assert engine.pool._pre_ping
    assert 0 < engine.pool._recycle <= 600
