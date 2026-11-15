"""The Question of the Day board tab: voter ID (U11), asking and moderation (U12), the tab (U13a).

Anyone can ask and vote without an account. One vote per browser: a random voter ID is kept in
the browser's localStorage. Streamlit can read cookies but not set them, so a few lines of
inline JS (Streamlit's built-in `st.components.v2`, no extra package) store the ID and send it
back. Clearing site data gives a new ID; the per-voter rate limits in `board_db` bound what that
can do.

Asking (U12): an exact repeat (ignoring case and punctuation) of a question on the board is a
vote for it; a similar one (fuzzy score >= 85) is shown to the asker first, because fuzzy text
matches also join different questions (sugar vs salt and blood pressure scores 92). A new
question gets one large-model check; the harness, not the model, decides whether it is shown.
"""

import logging
import os
import uuid
from typing import Literal

import streamlit as st
from pydantic import BaseModel
from rapidfuzz import fuzz
from sqlalchemy import Engine

from causal_genie.agent.router import normalize
from causal_genie.agent.state import Schema
from causal_genie.llm.client import LLMClient
from causal_genie.ui import board_db
from causal_genie.ui.live import passcode_ok

log = logging.getLogger(__name__)

SIMILAR = 85  # rapidfuzz token_set_ratio on normalized text

Problem = Literal["none", "abusive", "personal_info", "spam", "off_topic"]

NOT_CAUSAL = "not a cause-and-effect question"
PROBLEM_REASONS: dict[str, str] = {
    "abusive": "abusive or hateful language",
    "personal_info": "it seems to include personal information",
    "spam": "it looks like spam",
    "off_topic": "it isn't a research question",
}

_VOTER_JS = """
export default function (component) {
  const { data, setStateValue } = component;
  const KEY = "cg_voter_id";
  const ok = (v) => typeof v === "string" && /^[0-9a-f-]{36}$/.test(v);
  let id = null;
  try { id = window.localStorage.getItem(KEY); } catch (e) { id = null; }
  if (!ok(id)) {
    id = data;
    try { window.localStorage.setItem(KEY, id); } catch (e) { /* private mode: this visit only */ }
  }
  setStateValue("voter_id", id);
}
"""

_voter_component = st.components.v2.component("cg_voter_id", js=_VOTER_JS)


def valid_voter_id(value: object) -> str | None:
    """The ID if the browser sent a well-formed UUID, else None (the value is untrusted)."""
    if not isinstance(value, str) or len(value) != 36:
        return None
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return None
    return value if str(parsed) == value else None


def voter_id_candidate() -> str:
    """A fresh random ID for this session, which the browser keeps if it has none yet."""
    if "voter_id_candidate" not in st.session_state:
        st.session_state.voter_id_candidate = str(uuid.uuid4())
    return st.session_state.voter_id_candidate


def voter_id() -> str | None:
    """This browser's voter ID, or None until the browser has reported it (first render)."""
    result = _voter_component(
        key="cg_voter_id",
        data=voter_id_candidate(),
        default={"voter_id": None},
        on_voter_id_change=lambda: None,
        height=0,
    )
    return valid_voter_id(result.get("voter_id"))


class BoardCheck(Schema):
    """The model's labels for a posted question; the harness turns them into a status."""

    causal: bool  # asks whether something causes or changes something else
    problem: Problem  # the first problem that applies, or "none"


class Asked(BaseModel):
    """What happened to a posted question, with the message the asker sees."""

    outcome: Literal["shown", "hidden", "voted", "already_voted", "answered", "similar"]
    question_id: int
    message: str


def ask(db: Engine, text: str, voter_id: str, llm: LLMClient, *, force: bool = False) -> Asked:
    """Post a public question: a repeat becomes a vote, a new one is checked, then stored.

    `force` posts it even if a similar question exists (the asker chose "ask mine anyway").
    Raises `ValueError` for empty or over-long text and `board_db.LimitReached` at a limit.
    """
    text = text.strip()
    if not text or len(text) > board_db.MAX_CHARS:
        raise ValueError(f"A question must be 1–{board_db.MAX_CHARS} characters.")
    key = normalize(text)
    existing = board_db.questions_with_status(db, "shown", "answered")
    for q in existing:
        if q.normalized == key:
            return _repeat(db, q, voter_id)
    if not force:
        scored = [(fuzz.token_set_ratio(key, q.normalized), q) for q in existing]
        best = max(scored, key=lambda s: s[0], default=None)
        if best and best[0] >= SIMILAR:
            log.info("board: similar to question %d (score %.0f)", best[1].id, best[0])
            return Asked(
                outcome="similar",
                question_id=best[1].id,
                message="A similar question is already on the board.",
            )
    board_db.check_question_limit(db, voter_id)  # before paying for the model call
    check = llm.complete("large", "board_check", f"<question>\n{text}\n</question>", BoardCheck)
    qid = board_db.add_question(db, text, voter_id)
    reason = NOT_CAUSAL if not check.causal else PROBLEM_REASONS.get(check.problem)
    if reason:
        board_db.set_status(db, qid, "hidden", reason=reason)
        return Asked(
            outcome="hidden", question_id=qid, message=f"Not added to the board: {reason}."
        )
    board_db.set_status(db, qid, "shown")
    return Asked(outcome="shown", question_id=qid, message="Added to the board. Thanks!")


def _repeat(db: Engine, q: board_db.Question, voter_id: str) -> Asked:
    if q.status == "answered":
        return Asked(
            outcome="answered", question_id=q.id, message="The Genie has already answered this."
        )
    if board_db.vote(db, q.id, voter_id):
        return Asked(
            outcome="voted",
            question_id=q.id,
            message="This question is already on the board, so we counted it as your vote.",
        )
    return Asked(
        outcome="already_voted",
        question_id=q.id,
        message="This question is already on the board, and you have voted for it.",
    )


def _toggle(db: Engine, q: board_db.Question) -> None:
    if q.status == "hidden":
        board_db.set_status(db, q.id, "shown")
    else:
        board_db.set_status(db, q.id, "hidden", reason="hidden by admin")


def admin_panel(db: Engine) -> None:
    """Hide or unhide any open question, behind the live-run passcode (`CG_LIVE_PASSCODE`)."""
    entered = st.text_input("Admin passcode", type="password", key="board_admin_passcode")
    if not entered:
        return
    if not passcode_ok(entered, os.getenv("CG_LIVE_PASSCODE", "")):
        st.error("Wrong passcode.")
        return
    for q in board_db.questions_with_status(db, "pending", "shown", "hidden", "skipped"):
        left, right = st.columns([5, 1])
        reason = f" ({q.reason})" if q.reason else ""
        left.text(f"[{q.status}] {q.text}{reason}")  # plain text: never Markdown or HTML
        right.button(
            "Unhide" if q.status == "hidden" else "Hide",
            key=f"admin-{q.id}",
            on_click=_toggle,
            args=(db, q),
        )
