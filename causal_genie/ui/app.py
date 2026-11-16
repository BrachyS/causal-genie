"""Causal Genie in the browser: pick or type a question and see its saved run.

Shows recorded runs only (replay needs no keys); live mode comes with a passcode and spend
cap (T11.3). Run: `uv run streamlit run causal_genie/ui/app.py`. `CG_RUNS_DIR` picks the
folder of saved runs (default `gallery/`).

Two tabs: the Gallery is read-only (pick an answered question); "Ask & vote" is the one
place to type. There a question goes to the public board (`BOARD_DB_URL`, U13b) or, with
the passcode, gets a private live run now; the board to vote on is below. Links
`?run=<run_id>` and `?q=<question>` open a saved run in the Gallery (U14), so Bluesky posts
and shared links land on the answer.
"""

import logging
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
    dag_section,
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
from causal_genie.ui.view import (
    VERDICT_MEANINGS,
    Citation,
    anchor,
    citations,
    cited_html,
    gallery_runs,
    sources,
    verdict_guide,
)

log = logging.getLogger(__name__)

LOGO = Path(__file__).parent / "assets" / "genie.png"  # the Genie's lamp, also on Bluesky
CONTEXT = "context.md"  # a saved run's background note, written by hand for the gallery


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
            " · solid: backed by a cited source · dashed: assumed, no source"
        )
        st.graphviz_chart(dag_dot(state.edges))
        with st.expander("What backs each arrow"):
            st.markdown(body(dag_section(state)))

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
        st.download_button("Download report (Markdown)", state.report, f"{state.run_id}.md")


def header() -> None:
    """The brand: the Genie's lamp beside the name and the one-line promise."""
    lamp, title = st.columns([1, 9], vertical_alignment="center")
    lamp.image(str(LOGO), width=110)
    title.title("Causal Genie")
    title.caption(
        "Does X cause Y? Recorded runs: every number traces to a tool, every claim to a quote."
    )


TABS = ("📚 Gallery", "🙋 Ask & vote")

INTRO = f"""\
### How to use it

Ask a "does X cause Y?" question. The Genie reads the studies, draws a causal graph whose \
arrows cite their sources, finds public data and estimates the effect, or says plainly \
why it can't.

1. **{TABS[0]}:** pick an answered question and read the verdict, the graph, the estimate \
and the quoted sources.
2. **{TABS[1]}:** type your own question and post it to the board, or vote for someone \
else's. Each night the top-voted question is answered and joins the gallery.

**Reading an answer:** the verdict says how far the evidence goes, from an estimate from \
data down to "insufficient evidence". Every number traces to a tool or a quoted source, \
never to the model.
"""


def intro() -> None:
    """The sidebar note that tells a first-time visitor what the app does and where to go."""
    st.sidebar.markdown(INTRO)


def main() -> None:
    st.set_page_config(page_title="Causal Genie", page_icon=str(LOGO), layout="wide")
    header()
    intro()
    runs_dir = Path(os.getenv("CG_RUNS_DIR", str(GALLERY_DIR)))
    answers, ask = st.tabs(TABS)
    with answers:
        gallery(runs_dir)
    with ask:
        ask_and_vote(os.getenv("BOARD_DB_URL", ""), runs_dir)


def linked_label(entries: dict[str, tuple[str, Path]], runs_dir: Path) -> str | None:
    """The gallery entry a `?run=` or `?q=` link names, or None (with a warning if it names none).

    A run id only ever selects among the runs the gallery lists; it never builds a path that
    is read.
    """
    run_id, wanted = st.query_params.get("run"), st.query_params.get("q")
    if run_id:
        path = runs_dir / run_id / "state.json"
        for label, (_, listed) in entries.items():
            if board.RUN_ID.fullmatch(run_id) and listed == path:
                return label
        st.warning("That link's run isn't in the gallery; pick a question instead.")
    elif wanted:
        key = normalize(wanted[: board_db.MAX_CHARS])
        if matches := [label for label, (q, _) in entries.items() if normalize(q) == key]:
            return matches[-1]  # the latest run of that question
        st.warning("That link's question isn't in the gallery yet; ask it in the Ask & vote tab.")
    return None


def gallery(runs_dir: Path) -> None:
    """Read-only: pick an answered question and read the Genie's full answer."""
    st.caption("Questions the Genie has answered. Pick one to read the full answer.")
    entries = gallery_runs(runs_dir)
    if not entries:
        st.info("No saved runs to show yet.")
        return
    # Point the picker at a link's run once, so picking another one later still works.
    link_key = (st.query_params.get("run"), st.query_params.get("q"))
    label = linked_label(entries, runs_dir)
    if label and st.session_state.get("opened_link") != link_key:
        st.session_state["opened_link"] = link_key
        st.session_state["gallery_pick"] = label
    picked = st.selectbox("Answered question", list(entries), key="gallery_pick")
    path = entries[picked][1]
    log.info("showing run %s", path)
    context_note(path)
    show(RunState.load(path))


@st.cache_resource(show_spinner=False)
def board_engine(url: str) -> Engine:
    return board_db.connect(url)


def answered_link(question: str, runs_dir: Path) -> str | None:
    """A `?q=` link to the gallery run of this question, if there is one."""
    wanted = normalize(question)
    if any(normalize(q) == wanted for q, _ in gallery_runs(runs_dir).values()):
        return f"?q={quote(question)}"
    return None


def ask_and_vote(board_url: str, runs_dir: Path) -> None:
    """The one place to type a question: post it to the public board, or (with the passcode)
    get a private answer now. Below it, the board to vote on."""
    db: Engine | None = None
    if board_url:
        try:
            db = board_engine(board_url)
        except SQLAlchemyError:
            log.exception("board: database unavailable")
            st.warning("Board unavailable right now. The gallery still works.")
    passcode = os.getenv("CG_LIVE_PASSCODE", "")
    if db is None and not passcode:
        st.info("Asking is off in this demo. Read the answered questions in the Gallery tab.")
        return
    llm = None
    if db is not None and os.getenv("NEBIUS_API_KEY") and os.getenv("NEBIUS_BASE_URL"):
        llm = LLMClient(make_openai(), load_models())
    voter = board.voter_id() if db is not None else None

    st.subheader("Ask the Genie")
    if db is not None:
        st.caption(
            "Questions are public: anyone can vote, and the top-voted one is answered each "
            "night, then added to the Gallery."
        )
    with st.form("ask", clear_on_submit=True):
        text = st.text_input(
            "Your cause-and-effect question",
            max_chars=board_db.MAX_CHARS,
            placeholder="Does X cause Y?",
        )
        post = False
        if db is not None:
            post = st.form_submit_button(
                "Post to the board", type="primary", disabled=llm is None or voter is None
            )
        run, entered = False, ""
        if passcode:
            st.caption("Have a passcode? Get a private answer now instead (about 5 minutes).")
            entered = st.text_input("Passcode", type="password")
            run = st.form_submit_button("Run privately now")
    text = text.strip()
    if text and (post or run) and (link := answered_link(text, runs_dir)):
        st.info(f"That question is already answered: [read it in the Gallery]({link}).")
    elif text and post and db is not None and voter and llm:
        board.post_question(db, text, voter, llm)
    elif text and run:
        live_run(text, entered, passcode)
    if shown := st.session_state.get("live_shown"):
        with st.expander(f"Your private answer: {shown.question}", expanded=True):
            show(shown)

    if db is not None:
        st.divider()
        board.render_board(db, voter, llm, runs_dir)
        if passcode:
            with st.expander("Admin"):
                board.admin_panel(db)


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
