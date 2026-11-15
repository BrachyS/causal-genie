"""Causal Genie in the browser: pick or type a question and see its saved run.

Shows recorded runs only (replay needs no keys); live mode comes with a passcode and spend
cap (T11.3). Run: `uv run streamlit run causal_genie/ui/app.py`. `CG_RUNS_DIR` picks the
folder of saved runs (default `runs/`).
"""

import logging
import os
from pathlib import Path

import streamlit as st

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
from causal_genie.tools.render import dag_dot
from causal_genie.ui import live
from causal_genie.ui.gallery import GALLERY_DIR
from causal_genie.ui.live import passcode_ok
from causal_genie.ui.view import (
    VERDICT_MEANINGS,
    Citation,
    anchor,
    citations,
    cited_html,
    saved_runs,
    sources,
    verdict_guide,
)

log = logging.getLogger(__name__)


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


def main() -> None:
    st.set_page_config(page_title="Causal Genie", page_icon="🧞", layout="wide")
    st.title("🧞 Causal Genie")
    st.caption(
        "Does X cause Y? Recorded runs: every number traces to a tool, every claim to a quote."
    )

    runs = saved_runs(Path(os.getenv("CG_RUNS_DIR", str(GALLERY_DIR))))
    if not runs:
        st.info("No saved runs to show yet.")
        return
    picked = st.sidebar.radio("Gallery", list(runs))
    question = st.text_input("Your question", value=picked or "")
    by_key = {normalize(q): path for q, path in runs.items()}
    path = by_key.get(normalize(question))
    ran_live: dict[str, RunState] = st.session_state.setdefault("live_runs", {})
    if path is not None:
        log.info("showing run %s", path)
        show(RunState.load(path))
    elif normalize(question) in ran_live:
        show(ran_live[normalize(question)])
    else:
        live_section(question, ran_live)


def live_section(question: str, ran_live: dict[str, RunState]) -> None:
    """Offer a live run of a question with no saved run, behind the passcode and spend cap."""
    expected = os.getenv("CG_LIVE_PASSCODE", "")
    if not expected:
        st.info(
            "No saved run for this question. Live runs are off in this demo; "
            "pick one from the gallery."
        )
        return
    st.info("No saved run for this question. A live run takes about 5 minutes.")
    entered = st.text_input("Passcode for live runs", type="password")
    if not st.button("Run live"):
        return
    if not passcode_ok(entered, expected):
        st.error("Wrong passcode.")
        return
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
    show(state)


main()
