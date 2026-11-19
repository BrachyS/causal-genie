"""Storage for the Question of the Day board (U10): public questions and one vote per browser.

Any Postgres works through a connection string (the app's `st.connection("board", type="sql")`,
the nightly Action's `BOARD_DB_URL`); tests use SQLite. Queries are SQLAlchemy Core, no ORM, so
the same code runs on both.

Question text comes from strangers. It is stored and returned as data; callers escape it when
they show it and never put it in a shell command.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import (
    Column,
    DateTime,
    Engine,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError

from causal_genie.agent.router import normalize

log = logging.getLogger(__name__)

MAX_CHARS = 200
QUESTIONS_PER_DAY = 3  # per voter ID, rolling 24 hours
VOTES_PER_HOUR = 30  # per voter ID, rolling hour

Status = Literal["pending", "shown", "hidden", "running", "answered", "skipped"]
STATUSES: tuple[str, ...] = Status.__args__

metadata = MetaData()

questions = Table(
    "questions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("text", String(MAX_CHARS), nullable=False),
    Column("normalized", String(MAX_CHARS), nullable=False, index=True),
    Column("created_at", DateTime, nullable=False),
    Column("status", String(16), nullable=False, index=True),
    Column("reason", String(500)),
    Column("run_id", String(100)),
    Column("announcement", String(500)),
    Column("answered_at", DateTime),
    Column("voter_id", String(64), nullable=False, index=True),
)

votes = Table(
    "votes",
    metadata,
    Column("question_id", Integer, ForeignKey("questions.id"), nullable=False),
    Column("voter_id", String(64), nullable=False, index=True),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("question_id", "voter_id", name="one_vote_per_voter"),
)


class Question(BaseModel):
    """A board question as the app and the nightly job see it, with its vote count."""

    id: int
    text: str
    normalized: str
    created_at: datetime
    status: Status
    reason: str | None
    run_id: str | None
    announcement: str | None
    answered_at: datetime | None
    voter_id: str
    votes: int


class LimitReached(Exception):
    """A voter hit a rate limit; the message is shown to them as is."""


def now() -> datetime:
    """UTC time without a zone, stored the same way on SQLite and Postgres."""
    return datetime.now(UTC).replace(tzinfo=None)


def driver_url(url: str) -> str:
    """`url` with the psycopg (v3) driver named, as hosts hand out plain `postgresql://` URLs
    and SQLAlchemy would otherwise look for psycopg2, which is not installed."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


def connect(url: str) -> Engine:
    """An engine for `url` (e.g. `BOARD_DB_URL`) with the tables in place.

    The app keeps one engine for its lifetime, and a hosted Postgres closes connections left
    idle (e.g. while a 10-minute live run holds the script), so each connection is checked
    before use and renewed after 5 minutes rather than failing the next board query.
    """
    engine = create_engine(driver_url(url), pool_pre_ping=True, pool_recycle=300)
    create_tables(engine)
    return engine


def create_tables(engine: Engine) -> None:
    """Create the tables if they are missing; safe to call on every start."""
    metadata.create_all(engine, checkfirst=True)


def check_question_limit(engine: Engine, voter_id: str) -> None:
    """Raise `LimitReached` if this voter has asked their questions for the day."""
    since = now() - timedelta(days=1)
    if _count_since(engine, questions, voter_id, since) >= QUESTIONS_PER_DAY:
        raise LimitReached(
            f"You can ask {QUESTIONS_PER_DAY} questions a day. Please try again tomorrow."
        )


def add_question(engine: Engine, text: str, voter_id: str) -> int:
    """Store a new question as `pending` (moderation decides if it is shown) and return its id."""
    text = text.strip()
    if not text or len(text) > MAX_CHARS:
        raise ValueError(f"a question must be 1–{MAX_CHARS} characters, got {len(text)}")
    check_question_limit(engine, voter_id)
    row = {
        "text": text,
        "normalized": normalize(text),
        "created_at": now(),
        "status": "pending",
        "voter_id": voter_id,
    }
    with engine.begin() as conn:
        qid = conn.execute(questions.insert().values(row).returning(questions.c.id)).scalar_one()
    log.info("board: question %d added by voter %s", qid, voter_id)
    return qid


def vote(engine: Engine, question_id: int, voter_id: str) -> bool:
    """Count one vote on a shown question; a repeat by the same voter changes nothing.

    Returns True if the vote counted. With the question checked first, the only constraint a
    vote can break is one-vote-per-voter.
    """
    status = get_question(engine, question_id).status
    if status != "shown":
        raise ValueError(f"question {question_id} is {status}, not open for votes")
    if question_id in voted(engine, voter_id):
        return False
    if _count_since(engine, votes, voter_id, now() - timedelta(hours=1)) >= VOTES_PER_HOUR:
        raise LimitReached(f"You can cast {VOTES_PER_HOUR} votes an hour. Please try again later.")
    try:
        with engine.begin() as conn:
            conn.execute(
                votes.insert().values(question_id=question_id, voter_id=voter_id, created_at=now())
            )
    except IntegrityError:
        log.info("board: voter %s already voted on question %d", voter_id, question_id)
        return False
    return True


def _count_since(engine: Engine, table: Table, voter_id: str, since: datetime) -> int:
    query = (
        select(func.count())
        .select_from(table)
        .where(table.c.voter_id == voter_id, table.c.created_at > since)
    )
    with engine.connect() as conn:
        return conn.execute(query).scalar_one()


def voted(engine: Engine, voter_id: str) -> set[int]:
    """Ids of the questions this voter has voted on, for the "you voted" state."""
    with engine.connect() as conn:
        rows = conn.execute(select(votes.c.question_id).where(votes.c.voter_id == voter_id))
        return {r.question_id for r in rows}


def set_status(
    engine: Engine,
    question_id: int,
    status: Status,
    *,
    reason: str | None = None,
    run_id: str | None = None,
    announcement: str | None = None,
) -> None:
    """Move a question to `status`; `answered` also records its run and the time."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    values: dict[str, object] = {"status": status, "reason": reason}
    if run_id is not None:
        values["run_id"] = run_id
    if announcement is not None:
        values["announcement"] = announcement
    if status == "answered":
        values["answered_at"] = now()
    with engine.begin() as conn:
        result = conn.execute(
            questions.update().where(questions.c.id == question_id).values(values)
        )
    if result.rowcount == 0:
        raise KeyError(f"no question {question_id}")
    log.info("board: question %d -> %s (%s)", question_id, status, reason)


def _with_votes():
    counts = (
        select(votes.c.question_id, func.count().label("votes"))
        .group_by(votes.c.question_id)
        .subquery()
    )
    return select(questions, func.coalesce(counts.c.votes, 0).label("votes")).outerjoin(
        counts, counts.c.question_id == questions.c.id
    )


def _fetch(engine: Engine, query) -> list[Question]:
    with engine.connect() as conn:
        return [Question.model_validate(dict(r._mapping)) for r in conn.execute(query)]


def get_question(engine: Engine, question_id: int) -> Question:
    """One question by id, with its votes."""
    found = _fetch(engine, _with_votes().where(questions.c.id == question_id))
    if not found:
        raise KeyError(f"no question {question_id}")
    return found[0]


def questions_with_status(engine: Engine, *statuses: Status) -> list[Question]:
    """Questions in any of `statuses`, oldest first."""
    query = _with_votes().where(questions.c.status.in_(statuses)).order_by(questions.c.id)
    return _fetch(engine, query)


def board(engine: Engine) -> list[Question]:
    """Shown questions with their votes, oldest first; `picks.rank` orders them by votes."""
    return questions_with_status(engine, "shown")


def answered(engine: Engine) -> list[Question]:
    """Answered questions, newest answer first."""
    query = (
        _with_votes()
        .where(questions.c.status == "answered")
        .order_by(questions.c.answered_at.desc(), questions.c.id.desc())
    )
    return _fetch(engine, query)


def latest_answered(engine: Engine) -> Question | None:
    """Today's question: the most recently answered one, if any."""
    rows = answered(engine)
    return rows[0] if rows else None
