"""The nightly Question of the Day pick (U15, U16).

A rule picks the question, not a model: the shown question with the most votes, ties to the
oldest, among those on the board for at least a day (`MIN_AGE`), so every question gets a full
day of votes however late GitHub starts the job. Each night the GitHub Action runs
`python -m causal_genie.ui.picks run`: it answers up to `n` questions, exports each answer to
`gallery/` with a `pick.json`, and writes a PR body for the user to review. A run the model
couldn't finish, or a non-causal one, marks the question `skipped` and the next candidate gets
a try, at most 2 tries per slot. A run stopped by a service (LLM API, web search, HTTP) is not
the question's fault: the question goes back on the board and the night ends with the job
failed, since every other candidate would fail too. The question stays `running` until its PR
is merged (U18 marks it answered).

Question text comes from strangers. It only travels through Python and files here; the
workflow never puts it on a shell line.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import Engine

from causal_genie.agent.state import RunState, Schema, Verdict
from causal_genie.agent.steps.report import numbers
from causal_genie.llm.client import LLMClient, load_models, make_openai
from causal_genie.log import setup_logging
from causal_genie.ui import board_db
from causal_genie.ui.board_db import Question
from causal_genie.ui.gallery import GALLERY_DIR, export
from causal_genie.ui.live import run_live

log = logging.getLogger(__name__)

TRIES_PER_SLOT = 2
MIN_AGE = timedelta(hours=24)  # a question waits this long on the board before it can be picked
MAX_N = 3
TITLE_CHARS = 100

Answer = Callable[[str, Path], RunState]


class Pick(BaseModel):
    """One answered board question, as saved in `gallery/<run_id>/pick.json`."""

    question_id: int
    question: str
    votes: int
    run_id: str
    verdict: Verdict | None
    announcement: str  # the Genie's line, or the plain "Up for the Genie: …"
    genie: bool  # True if `announcement` is the Genie's line (not posted on sensitive topics)


class Nightly(BaseModel):
    """What one nightly job did: its picks, and whether any slot ended empty."""

    picks: list[Pick]
    failed: bool


def rank(questions: list[Question]) -> list[Question]:
    """Shown questions, most votes first, ties to the oldest."""
    shown = [q for q in questions if q.status == "shown"]
    return sorted(shown, key=lambda q: (-q.votes, q.created_at, q.id))


class PickLine(Schema):
    """The model's draft announcement and its read of the topic; the harness decides."""

    sensitive: bool  # health risks people act on, mental health, harm, groups of people, etc.
    line: str  # one playful sentence, no numbers


PLAIN_PREFIX = "Up for the Genie: "


def plain_announcement(q: Question) -> str:
    return f"{PLAIN_PREFIX}{q.text}"


def genie_line(question: str, llm: LLMClient) -> str | None:
    """The Genie's line announcing a picked question, or None when the gate forbids one."""
    judged = llm.complete("large", "genie_pick", f"<question>\n{question}\n</question>", PickLine)
    line = judged.line.strip()
    log.info("genie_pick: sensitive=%s line=%r", judged.sensitive, line)
    if judged.sensitive:
        reason = "sensitive topic"
    elif numbers(line) - numbers(question):
        reason = "the line contains a number not in the question"
    elif not line:
        reason = "empty line"
    else:
        return line
    log.info("genie_pick: plain announcement: %s", reason)
    return None


def no_genie(q: Question) -> str | None:
    return None


def service_failure(state: RunState) -> bool:
    """True if a service stopped the run (API, HTTP, search), not the model's output."""
    return state.failure is not None and not state.failure.startswith("ModelOutputError")


def skip_reason(state: RunState) -> str | None:
    """Why a run can't be tonight's answer, or None if it can."""
    if state.failure is not None:
        return f"partial run: {state.failure}"
    if state.verdict == "not_causal":
        return "not a cause-and-effect question"
    return None


def nightly(
    db: Engine,
    n: int,
    answer: Answer,
    runs_dir: Path,
    gallery_dir: Path,
    announce: Callable[[Question], str | None] = no_genie,
) -> Nightly:
    """Answer up to `n` top questions at least `MIN_AGE` old, falling back to the next candidate
    when a run fails.

    `announce` drafts the Genie's line for a pick, None for the plain announcement.
    """
    cutoff = board_db.now() - MIN_AGE
    candidates = [q for q in rank(board_db.board(db)) if q.created_at <= cutoff]
    if not candidates:
        log.info("nightly: no shown question is at least %s old; nothing to answer", MIN_AGE)
    picks: list[Pick] = []
    failed = False
    for slot in range(n):
        if not candidates:
            break
        answered = False
        for _ in range(min(TRIES_PER_SLOT, len(candidates))):
            q = candidates.pop(0)
            board_db.set_status(db, q.id, "running")
            try:
                state = answer(q.text, runs_dir)
            except Exception:
                log.exception("nightly: run for question %d crashed", q.id)
                board_db.set_status(db, q.id, "shown", reason="run crashed; back on the board")
                raise
            if service_failure(state):
                log.error("nightly: question %d hit a service failure: %s", q.id, state.failure)
                board_db.set_status(
                    db, q.id, "shown", reason=f"service failure, back on the board: {state.failure}"
                )
                return Nightly(picks=picks, failed=True)
            reason = skip_reason(state)
            if reason:
                log.warning("nightly: question %d skipped: %s", q.id, reason)
                board_db.set_status(db, q.id, "skipped", reason=reason, run_id=state.run_id)
                continue
            line = announce(q)
            pick = Pick(
                question_id=q.id,
                question=q.text,
                votes=q.votes,
                run_id=state.run_id,
                verdict=state.verdict,
                announcement=line or plain_announcement(q),
                genie=line is not None,
            )
            export(runs_dir / state.run_id, gallery_dir)
            (gallery_dir / state.run_id / "pick.json").write_text(pick.model_dump_json(indent=2))
            board_db.set_status(db, q.id, "running", run_id=state.run_id)
            log.info("nightly: slot %d -> question %d, run %s", slot + 1, q.id, state.run_id)
            picks.append(pick)
            answered = True
            break
        if not answered:
            log.error("nightly: slot %d has no answer after %d tries", slot + 1, TRIES_PER_SLOT)
            failed = True
    return Nightly(picks=picks, failed=failed)


def _inert(text: str) -> str:
    """Stranger text for a fenced block: no backticks, so it can't close the fence."""
    return text.replace("`", "'")


def pr_title(night: Nightly, day: str) -> str:
    """The review PR's title: the date and the top pick's question on one line, cut to
    TITLE_CHARS so the reviewer sees what was answered. Titles don't render Markdown."""
    head = f"Question of the Day {day}: "
    more = f" (+{len(night.picks) - 1} more)" if len(night.picks) > 1 else ""
    question = " ".join(night.picks[0].question.split())
    room = TITLE_CHARS - len(head) - len(more)
    if len(question) > room:
        question = question[: room - 1].rstrip() + "…"
    return head + question + more


def pr_body(night: Nightly) -> str:
    """The review PR's description. Question text sits in fenced blocks, where Markdown,
    links and @mentions do nothing."""
    parts = ["Question of the Day answers to review. Once merged, they post one a day (U18).", ""]
    for p in night.picks:
        votes = f"{p.votes} vote" + ("" if p.votes == 1 else "s")
        parts += [
            f"### Run `{p.run_id}`: question {p.question_id}, {votes}, verdict `{p.verdict}`",
            "",
            "```text",
            _inert(p.question),
            "```",
            "",
            "Genie's line:",
            "",
            "```text",
            _inert(p.announcement),
            "```",
            "",
        ]
    return "\n".join(parts)


def _n(value: str) -> int:
    n = int(value)
    if not 1 <= n <= MAX_N:
        raise argparse.ArgumentTypeError(f"n must be 1–{MAX_N}")
    return n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Answer the board's top questions.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="answer up to n questions and write the PR body")
    run.add_argument("--n", type=_n, default=1)
    run.add_argument("--runs", type=Path, default=Path("runs"))
    run.add_argument("--gallery", type=Path, default=GALLERY_DIR)
    run.add_argument("--body-file", type=Path, required=True)
    run.add_argument("--title-file", type=Path, required=True)
    args = parser.parse_args(argv)
    setup_logging()
    db = board_db.connect(os.environ["BOARD_DB_URL"])
    genie = os.getenv("GENIE_MODE", "").lower() in ("1", "true", "yes", "on")
    llm = LLMClient(make_openai(), load_models()) if genie else None

    def announce(q: Question) -> str | None:
        return genie_line(q.text, llm) if llm else None

    night = nightly(db, args.n, run_live, args.runs, args.gallery, announce)
    if night.picks:
        args.body_file.write_text(pr_body(night))
        args.title_file.write_text(pr_title(night, datetime.now(UTC).strftime("%Y-%m-%d")))
    if night.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
