"""Causal Genie in the browser: ask a question, vote, and browse the answered ones.

Shows recorded runs only (replay needs no keys); live mode comes with a passcode and spend
cap (T11.3). Run: `uv run streamlit run causal_genie/ui/app.py`. `CG_RUNS_DIR` picks the
folder of saved runs (default `gallery/`).

One page. On top, the one place to type: a question goes to the public board
(`BOARD_DB_URL`, U13b) or, with the passcode, gets a private live run now. Below, the
gallery: a card per answered question, in a band per topic. The board to vote on is in the
sidebar. A card opens its full answer on its own page, `?run=<run_id>`; that link and
`?q=<question>` are also how Bluesky posts and shared links land on an answer (U14).
"""

import logging
import math
import os
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from causal_genie.agent.certainty import evidence_flow
from causal_genie.agent.router import normalize
from causal_genie.agent.state import RunState, Sentence
from causal_genie.agent.steps.report import (
    DISCLAIMER,
    FINDINGS,
    VERDICT_LABELS,
    data_section,
    estimate_section,
    evidence_section,
    ledger_entries,
    question_section,
)
from causal_genie.llm.client import LLMClient, load_models, make_openai
from causal_genie.tools.render import dag_dot
from causal_genie.ui import board, board_db, live
from causal_genie.ui.gallery import GALLERY_DIR
from causal_genie.ui.live import passcode_ok
from causal_genie.ui.pdf import report_pdf
from causal_genie.ui.view import (
    PER_PAGE,
    VERDICT_MEANINGS,
    Card,
    Citation,
    anchor,
    answered_on,
    bands,
    category,
    citations,
    cited_html,
    dag_backing,
    escape,
    gallery_cards,
    gallery_runs,
    matching,
    sources,
    verdict_choices,
    verdict_guide,
    verdict_list,
    with_verdicts,
)

log = logging.getLogger(__name__)

LOGO = Path(__file__).parent / "assets" / "genie.png"  # the Genie's lamp, also on Bluesky
CONTEXT = "context.md"  # a saved run's background note, written by hand for the gallery
CARD_HEIGHT = 330  # px: every card in a band the same size; a long one scrolls inside


def body(section: list[str]) -> str:
    """A report section's Markdown without its `## ` heading (the app adds its own)."""
    return "\n".join(line for line in section if not line.startswith("## "))


def cited(sentences: list[Sentence], entries: dict[str, str], cites: dict[str, Citation]) -> str:
    return " ".join(cited_html(s, entries, cites) for s in sentences)


def prose(
    title: str, sentences: list[Sentence], entries: dict[str, str], cites: dict[str, Citation]
) -> None:
    if sentences:
        st.subheader(title)
        st.markdown(cited(sentences, entries, cites), unsafe_allow_html=True)


def short_answer(state: RunState, entries: dict[str, str], cites: dict[str, Citation]) -> None:
    """The verdict, what it means, and the summary's first sentence, which states the answer."""
    assert state.verdict is not None
    with st.container(border=True):
        st.markdown("#### Short answer")
        st.markdown(f"**{VERDICT_LABELS[state.verdict]}** — {VERDICT_MEANINGS[state.verdict]}")
        if state.prose is not None and state.prose.summary:
            st.markdown(cited(state.prose.summary[:1], entries, cites), unsafe_allow_html=True)
        with st.expander("What the verdicts mean"):
            st.markdown(verdict_guide(state.verdict))


def context_note(state_path: Path) -> None:
    """A gallery run's background (`context.md` next to its `state.json`), if it has one.

    Written by the repo owner, never by visitors, so it is shown as Markdown.
    """
    note = state_path.parent / CONTEXT
    if note.is_file():
        st.info(note.read_text(), icon="📋")


def show(state: RunState) -> None:
    """The run's verdict, prose with clickable numbers and citations, DAG, evidence and data."""
    if state.failure is not None:
        st.warning(
            f"Partial run: the `{state.step}` step failed, so no verdict was reached. "
            "This is not a finding that no evidence exists."
        )
    entries, cites = ledger_entries(state), citations(state)
    if state.failure is None and state.verdict is not None:
        short_answer(state, entries, cites)
    if state.verdict == "not_causal" and state.route is not None:
        st.markdown(state.route.reason)

    draft = state.prose
    if draft is not None:
        prose(FINDINGS, draft.summary[1:], entries, cites)
    if state.genie:
        st.info(f"🧞 Genie's aside: {state.genie}")

    if state.rounds or state.studies:
        flow = evidence_flow(state.studies)
        found, deduped, included, level = st.columns(4)
        found.metric("Sources found", flow.found)
        deduped.metric("Studies after dedupe", flow.after_dedupe)
        included.metric("Included", flow.included)
        if state.certainty is not None:
            level.metric("Certainty", state.certainty.level.replace("_", " "))

    if state.edges:
        st.subheader("Causal graph")
        st.caption(
            "Purple arrow with ?: the question being tested (the answer is above, not the arrow)"
            " · solid: backed by a cited source · dashed: assumed, no source."
            " Boxes: purple = the question's cause and effect · green = from the research"
            " (on at least one cited arrow) · grey, dashed = the model's own idea, no source"
        )
        st.graphviz_chart(dag_dot(state.edges))
        with st.expander("What backs each arrow"):
            st.markdown(dag_backing(state), unsafe_allow_html=True)

    if state.verdict != "not_causal" and (state.results or state.design is not None):
        st.subheader("Estimate")
        st.markdown(body(estimate_section(state)))
    if draft is not None:
        prose("Comparison with published", draft.comparison, entries, cites)
        prose("Limitations", draft.limitations, entries, cites)

    with st.expander("Question (PECO)"):
        st.markdown(body(question_section(state)))
    if evidence := evidence_section(state):
        with st.expander("Evidence"):
            st.markdown(body(evidence))
    if data := data_section(state):
        with st.expander("Data used"):
            st.markdown(body(data))

    if cards := sources(state):
        st.subheader("Sources")
        for card in cards:
            st.markdown(f'<div id="{anchor(card.key)}"></div>', unsafe_allow_html=True)
            st.markdown(f"**{card.label}** · {card.markdown}")

    st.divider()
    st.caption(DISCLAIMER)
    tokens = state.tokens
    st.caption(f"Run `{state.run_id}` · {tokens['in']:,} tokens in / {tokens['out']:,} out")
    if state.report:
        st.download_button(
            "Download report (PDF)",
            pdf_of(state.report, answered_on(state.run_id)),
            f"{state.run_id}.pdf",
            mime="application/pdf",
        )


@st.cache_data(show_spinner=False, max_entries=50)
def pdf_of(report: str, answered: str) -> bytes:
    """The report's PDF, dated, built once per report while the app runs."""
    return report_pdf(report, answered)


TAGLINE = (
    "**Does X really cause Y?** The Genie reads the studies, maps what else could explain "
    "the link, and runs the numbers on real data, then tells you plainly how sure anyone "
    "can be."
)
PROOF = "Every number traces to a tested tool, every claim to a quote from its source."


def header() -> None:
    """The brand: the Genie's lamp beside the name and the one-line promise."""
    # The lamp spans the whole text block: title, tagline and proof line (~170 px tall).
    lamp, title = st.columns([2, 11], vertical_alignment="center")
    lamp.image(str(LOGO), width=170)
    title.title("Causal Genie")
    title.markdown(TAGLINE)
    title.caption(PROOF)


INTRO = """\
### How to use it

1. **Ask** a "does X cause Y?" question in the box at the top of the page. It goes on the \
public board below, where anyone can vote.
2. **Vote** for the question you want answered next. The Genie answers up to one question a \
day: the top-voted one asked at least a day ago, when there is one. It then joins the gallery.
3. **Browse** the answered questions by topic, and open any card for the full answer: the \
graph, the estimate and the quoted sources.

**Reading an answer:** the verdict says how far the evidence goes, from an estimate from \
data down to "insufficient evidence". Every number traces to a tool or a quoted source, \
never to the model.
"""


VOTE_BANNER = """\
<div style="background: linear-gradient(135deg, #7c3aed, #db2777); color: #fff;
  border-radius: 14px; padding: 14px 16px; margin-bottom: 4px;
  box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);">
  <div style="font-size: 1.25rem; font-weight: 700;">🗳️ Vote on what's next</div>
  <div style="font-size: 0.9rem; opacity: 0.92; margin-top: 2px;">
    The Genie answers up to one top-voted question a day. One tap, no sign-up.
  </div>
</div>
"""


def intro() -> None:
    """The sidebar's reference notes, below the vote: how to use the app (folded) and every
    verdict's name, each opening to what it means."""
    with st.sidebar.expander("How to use it"):
        st.markdown(INTRO)
    st.sidebar.markdown("#### What the verdicts mean")
    st.sidebar.markdown(verdict_list(), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Causal Genie", page_icon=str(LOGO), layout="wide")
    header()
    vote_slot = st.sidebar.container()  # on top of the sidebar, filled last (see below)
    intro()
    runs_dir = Path(os.getenv("CG_RUNS_DIR", str(GALLERY_DIR)))
    db = board_or_none(os.getenv("BOARD_DB_URL", ""))
    llm = None
    if db is not None and os.getenv("NEBIUS_API_KEY") and os.getenv("NEBIUS_BASE_URL"):
        llm = LLMClient(make_openai(), load_models())
    voter = board.voter_id() if db is not None else None
    passcode = os.getenv("CG_LIVE_PASSCODE", "")

    entries = gallery_runs(runs_dir)
    if label := linked_label(entries):
        answer_page(*entries[label])
    else:
        ask_box(db, voter, llm, passcode, runs_dir)
        st.divider()
        gallery(runs_dir)

    if db is not None:  # after the ask box, so a question posted on this click is listed
        with vote_slot:
            st.markdown(VOTE_BANNER, unsafe_allow_html=True)
            board.render_board(db, voter, llm, runs_dir)
            if passcode:
                with st.expander("Admin"):
                    board.admin_panel(db)


def linked_label(entries: dict[str, tuple[str, Path]]) -> str | None:
    """The gallery entry a `?run=` or `?q=` link names, or None (with a warning if it names none).

    A run id only ever selects among the runs the gallery lists; it never builds a path that
    is read.
    """
    run_id, wanted = st.query_params.get("run"), st.query_params.get("q")
    if run_id:
        for label, (_, listed) in entries.items():
            if board.RUN_ID.fullmatch(run_id) and listed.parent.name == run_id:
                return label
        st.warning("That link's run isn't in the gallery; pick a question below instead.")
    elif wanted:
        key = normalize(wanted[: board_db.MAX_CHARS])
        if matches := [label for label, (q, _) in entries.items() if normalize(q) == key]:
            return matches[-1]  # the latest run of that question
        st.warning("That link's question isn't in the gallery yet; ask it in the box below.")
    return None


def open_run(run_id: str) -> None:
    """Show the full answer of `run_id` on its own page, a link that can be shared."""
    st.query_params.clear()
    st.query_params["run"] = run_id


def answer_page(question: str, path: Path) -> None:
    """One run's full answer, with the way back to the gallery."""
    st.button("← All questions", key="back", on_click=st.query_params.clear)
    st.header(escape(question))
    if day := answered_on(path.parent.name):
        st.caption(
            f"Answered {day}, from the evidence found then. If new evidence comes out, the "
            "question can be re-run, and the newest answer replaces this one."
        )
    log.info("showing run %s", path)
    context_note(path)
    show(RunState.load(path))
    st.button("← All questions", key="back-bottom", on_click=st.query_params.clear)


def turn(key: str, step: int) -> None:
    st.session_state[key] = st.session_state.get(key, 0) + step


def card_view(c: Card) -> None:
    """A run at a glance: question, verdict, short answer, the Genie's aside, and the way in."""
    with st.container(border=True, height=CARD_HEIGHT):
        st.markdown(f"**{escape(c.question)}**")
        details = [escape(c.tag)] if c.tag else []
        if c.answered:
            details.append(f"Answered {c.answered}")
        if details:
            st.caption(" · ".join(details))
        st.markdown(c.verdict)
        if c.answer:
            st.markdown(c.answer)
        if c.genie:
            st.markdown(f"🧞 *{c.genie}*")
    st.button(
        "Read full answer →",
        key=f"open-{c.run_id}",
        on_click=open_run,
        args=(c.run_id,),
        use_container_width=True,
    )


def band(name: str, cards: list[Card]) -> None:
    """One category: a row of cards, with arrows to page left and right when there are more."""
    key = f"page-{name}"
    pages = math.ceil(len(cards) / PER_PAGE)
    page = min(st.session_state.get(key, 0), pages - 1)
    heading, back, forward = st.columns([12, 1, 1], vertical_alignment="bottom")
    heading.subheader(name)
    if pages > 1:
        back.button("◀", key=f"prev-{name}", disabled=page == 0, on_click=turn, args=(key, -1))
        forward.button(
            "▶", key=f"next-{name}", disabled=page == pages - 1, on_click=turn, args=(key, 1)
        )
    shown = cards[page * PER_PAGE :][:PER_PAGE]
    for column, c in zip(st.columns(PER_PAGE)[: len(shown)], shown, strict=True):
        with column:
            card_view(c)


def gallery(runs_dir: Path) -> None:
    """Read-only: the answered questions as cards, one band per topic."""
    heading, search, verdicts = st.columns([3, 2, 2], vertical_alignment="bottom")
    heading.subheader("📚 Answered questions")
    query = search.text_input(
        "Search answered questions",
        key="search",
        placeholder="🔍 Search answered questions",
        label_visibility="collapsed",
    )
    cards = gallery_cards(runs_dir)
    picked = verdicts.multiselect(
        "Filter by verdict",
        verdict_choices(cards),
        key="verdicts",
        placeholder="Filter by verdict",
        label_visibility="collapsed",
    )
    if not cards:
        st.info("No saved runs to show yet.")
        return
    cards = with_verdicts(matching(cards, query), picked)
    if not cards:
        st.info("No answered question matches. Ask it in the box above!")
        return
    llm = topic_llm()
    for name, in_band in bands(cards, lambda q: topic_of(q, llm)):
        band(name, in_band)


@st.cache_resource(show_spinner=False)
def topic_llm() -> LLMClient | None:
    """The small model that files new gallery questions into a band, if keys are set."""
    if os.getenv("NEBIUS_API_KEY") and os.getenv("NEBIUS_BASE_URL"):
        return LLMClient(make_openai(), load_models())
    return None


@st.cache_data(show_spinner=False)
def topic_of(question: str, _llm: LLMClient | None) -> str:
    """A question's band, picked once per question while the app runs (one small call)."""
    return category(question, _llm)


@st.cache_resource(show_spinner=False)
def board_engine(url: str) -> Engine:
    return board_db.connect(url)


def board_or_none(url: str) -> Engine | None:
    """The board's database, or None if there is none or it is down (the gallery still works)."""
    if not url:
        return None
    try:
        return board_engine(url)
    except SQLAlchemyError:
        log.exception("board: database unavailable")
        st.warning("Board unavailable right now. The gallery still works.")
        return None


def answered_link(question: str, runs_dir: Path) -> str | None:
    """A `?q=` link to the gallery run of this question, if there is one."""
    wanted = normalize(question)
    if any(normalize(q) == wanted for q, _ in gallery_runs(runs_dir).values()):
        return f"?q={quote(question)}"
    return None


CHATBOT_NOTE = """\
Try your question on any chatbot you like and compare. Here's what to look for.

- **Every source shown and checked.** A chatbot often names a few studies it remembers and \
thinks are representative. It can't show you where in a study a claim comes from, and its \
citations are sometimes wrong. The Genie searches fresh, shows every source it used, and keeps \
a claim only if it finds a quote that backs it up on the source page.
- **Recent evidence.** A chatbot that answers from memory only knows what was published before \
it was trained. The Genie searches for new studies every time and shows the date of each answer.
- **Only real studies count.** The Genie throws out retracted papers. Ten news stories about \
the same study count as one study.
- **Numbers are calculated, not remembered.** A number from a chatbot can be off, or come from \
one small study and sound like the whole picture. The Genie computes effect sizes with tested \
statistics code, from the raw data when it can get it, and shows where each number comes from.
- **Confidence follows fixed rules.** The Genie bases its confidence on how the studies were \
designed and whether they agree. When studies disagree, it explains why instead of counting \
votes.
- **You can check its work.** Every run is saved with all its searches and sources.

**When a chatbot is fine:** quick facts, definitions, or questions that aren't about cause and \
effect. A Genie answer takes several minutes, so save it for "does X cause Y?" questions where \
you want to see the evidence.

*The Genie uses AI models too, to find and read evidence. The checks, the math and the final \
verdict come from tested code.*
"""


def chatbot_note() -> None:
    """How the Genie differs from asking a chatbot, collapsed under the ask box."""
    with st.expander("How is this different from asking a chatbot?"):
        st.markdown(CHATBOT_NOTE)


def ask_box(
    db: Engine | None, voter: str | None, llm: LLMClient | None, passcode: str, runs_dir: Path
) -> None:
    """The one place to type a question, on top of the page: post it to the public board, or
    (with the passcode) get a private answer now."""
    st.subheader("Ask the Genie 🧞")
    if db is None and not passcode:
        st.info("Asking is off in this demo. Browse the answered questions below.")
        chatbot_note()
        return
    if db is not None:
        st.caption(
            "Questions are public: anyone can vote in the sidebar. The Genie answers up to one "
            "question a day, the top-voted one when there is one, then adds it to the gallery "
            "below."
        )
    with st.form("ask", clear_on_submit=True, border=False):
        field, submit = st.columns([5, 1], vertical_alignment="bottom")
        text = field.text_input(
            "Your cause-and-effect question",
            max_chars=board_db.MAX_CHARS,
            placeholder="Does X cause Y? e.g. Does coffee make you live longer?",
        )
        post = False
        if db is not None:
            post = submit.form_submit_button(
                "Post to the board",
                type="primary",
                disabled=llm is None or voter is None,
                use_container_width=True,
            )
        run, entered = False, ""
        if passcode:
            with st.expander("Have a passcode? Get a private answer now (about 5 to 10 minutes)"):
                entered = st.text_input("Passcode", type="password")
                run = st.form_submit_button("Run privately now")
    text = text.strip()
    if text and (post or run) and (link := answered_link(text, runs_dir)):
        st.info(f"That question is already answered: [read the answer]({link}).")
    elif text and post and db is not None and voter and llm:
        board.post_question(db, text, voter, llm)
    elif text and run:
        live_run(text, entered, passcode)
    if db is not None:
        board.ask_outcome(db, voter, llm)
    chatbot_note()
    if shown := st.session_state.get("live_shown"):
        with st.expander(f"Your private answer: {shown.question}", expanded=True):
            show(shown)


def live_run(question: str, entered: str, expected: str) -> None:
    """A private live run behind the passcode and the daily spend cap, one at a time."""
    if not passcode_ok(entered, expected):
        st.error("Wrong passcode.")
        return
    ran_live: dict[str, RunState] = st.session_state.setdefault("live_runs", {})
    if (state := ran_live.get(normalize(question))) is None:
        if not live.RUN_LOCK.acquire(blocking=False):
            st.warning("Another live run is going; try again in a few minutes.")
            return
        try:
            spend, day = live.ledger(), live.today()
            if refusal := spend.refusal(day):
                st.warning(refusal)
                return
            spend.record(day, tokens=0)  # count the run before it spends anything
            with st.spinner("Researching, building the DAG, finding data…"):
                state = live.run_live(question, live.live_dir())
            spend.record(day, tokens=sum(state.tokens.values()), runs=0)
        finally:
            live.RUN_LOCK.release()
        ran_live[normalize(question)] = state
    st.session_state["live_shown"] = state


main()
