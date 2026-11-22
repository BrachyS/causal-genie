"""Streamlit UI (T11.1): saved-run lookup, cited prose with clickable numbers, source cards,
gallery cards in category bands, and the app itself (Streamlit's AppTest, no browser, no
network)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from causal_genie.agent.state import DagEdge, ReportDraft, Result, RunState, Sentence
from causal_genie.agent.steps.report import VERDICT_LABELS, ledger_entries, write_report
from causal_genie.tools.stats import risk_difference
from causal_genie.ui import board, board_db
from causal_genie.ui.view import (
    CATEGORIES,
    GENIE_CHARS,
    OTHER,
    PER_PAGE,
    TOPICS,
    VERDICT_MEANINGS,
    Card,
    anchor,
    answered_on,
    bands,
    card,
    category,
    citations,
    cited_html,
    dag_backing,
    gallery_cards,
    gallery_runs,
    matching,
    saved_runs,
    sources,
    verdict_choices,
    verdict_guide,
    verdict_list,
    with_verdicts,
)

APP = Path(__file__).parents[1] / "causal_genie" / "ui" / "app.py"


@pytest.fixture
def ui_state(sample_state: RunState) -> RunState:
    """The sample D1 run, estimated and reported, with prose citing a claim and a result."""
    outputs = risk_difference(22, 117, 42, 116) | {
        "rows_excluded_missing": 0,
        "mapped_from": {"licorice_gargle": "treat", "sore_throat": "throatPain > 0"},
    }
    sample_state.results = [
        Result(id="res1", tool="rct_effect", tool_version="0.1.0", inputs_hash="h", outputs=outputs)
    ]
    sample_state.verdict = "estimated"
    sample_state.prose = ReportDraft(
        summary=[
            Sentence(text="At 30 min, risk fell by 17.4 percentage points.", cites=["res1"]),
            Sentence(text="A trial found the gargle reduced sore throat.", cites=["c1", "F-flow"]),
        ],
        comparison=[],
        limitations=[],
    )
    sample_state.report = write_report(sample_state)
    return sample_state


def save(state: RunState, runs_dir: Path, run_id: str) -> Path:
    return state.model_copy(update={"run_id": run_id}).save(runs_dir)


def test_saved_runs_keeps_the_latest_finished_run_per_question(ui_state, tmp_path) -> None:
    save(ui_state, tmp_path, "20261101-000000-aaaaaa")
    latest = save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    unfinished = ui_state.model_copy(update={"report": None, "question": "Unfinished?"})
    save(unfinished, tmp_path, "20261103-000000-cccccc")
    (tmp_path / "no-state").mkdir()

    assert saved_runs(tmp_path) == {ui_state.question: latest}


def test_saved_runs_skip_and_log_runs_of_an_older_schema(ui_state, tmp_path, step_logs) -> None:
    kept = save(ui_state, tmp_path, "20261101-000000-aaaaaa")
    old = tmp_path / "20261102-000000-bbbbbb" / "state.json"
    old.parent.mkdir()
    old.write_text(kept.read_text().replace('"rounds":', '"plans": [], "rounds":', 1))

    assert saved_runs(tmp_path) == {ui_state.question: kept}
    assert "20261102-000000-bbbbbb" in step_logs.text


@pytest.fixture
def two_studies(ui_state: RunState) -> RunState:
    """`ui_state` plus a second study, and a second claim from the first study."""
    other = ui_state.studies[0].model_copy(
        update={"id": "s2", "first_author": "Agarwal", "year": 2009}
    )
    ui_state.studies.append(other)
    ui_state.claims += [
        ui_state.claims[0].model_copy(update={"id": "c2", "study_id": "s2"}),
        ui_state.claims[0].model_copy(update={"id": "c3", "text": "236 adults"}),
    ]
    assert ui_state.prose is not None
    ui_state.prose.summary.append(Sentence(text="Two trials agree.", cites=["c2", "c3", "c1"]))
    return ui_state


def _with_dag(state: RunState) -> RunState:
    """`state` with a question arrow, an assumed arrow, and an arrow backed by a study the
    prose never cites (s3)."""
    third = state.studies[0].model_copy(update={"id": "s3", "first_author": "Lee", "year": 2020})
    state.studies.append(third)
    state.claims.append(state.claims[0].model_copy(update={"id": "c9", "study_id": "s3"}))
    state.edges = [
        DagEdge(source="licorice_gargle", target="sore_throat", claim_ids=[], question=True),
        DagEdge(source="age", target="sore_throat", claim_ids=["c9", "c1", "unknown"]),
        DagEdge(source="age", target="licorice_gargle", claim_ids=[]),
    ]
    state.exogenous = ["age"]
    return state


def test_studies_backing_only_an_arrow_are_numbered_after_the_prose_ones(two_studies) -> None:
    cites = citations(_with_dag(two_studies))

    assert (cites["c9"].key, cites["c9"].label) == ("s3", "Study 3")
    assert cites["c1"].label == "Study 1"  # the prose numbering is unchanged
    assert [c.label for c in sources(two_studies)][-1] == "Study 3"


def test_dag_backing_links_each_arrow_to_its_source_cards(two_studies) -> None:
    md = dag_backing(_with_dag(two_studies))

    lines = md.splitlines()
    assert lines[0] == "- licorice\\_gargle → sore\\_throat: the question being tested"
    assert lines[1] == (
        '- age → sore\\_throat: [<a href="#cite-s3">Study 3</a>] '
        '[<a href="#cite-s1">Study 1</a>] unknown'
    )
    assert lines[2] == "- age → licorice\\_gargle: assumed (no source)"
    assert "Treated as exogenous" in md and "c9" not in md


def test_citations_number_studies_and_analyses_in_citation_order(two_studies) -> None:
    cites = citations(two_studies)

    assert {k: (c.key, c.label) for k, c in cites.items()} == {
        "res1": ("res1", "Analysis 1"),
        "c1": ("s1", "Study 1"),
        "c2": ("s2", "Study 2"),
        "c3": ("s1", "Study 1"),  # a second claim from the same study keeps its number
    }


def test_cited_html_links_numbers_to_the_entry_that_backs_them(ui_state) -> None:
    html = cited_html(ui_state.prose.summary[0], ledger_entries(ui_state), citations(ui_state))

    # 17.4 is in the tool result; 30 comes from the question (PECO), so it is not linked
    assert f'<a href="#{anchor("res1")}">17.4</a>' in html
    assert "At 30 min" in html
    assert f'[<a href="#{anchor("res1")}">Analysis 1</a>]' in html
    assert "res1]" not in html


def test_cited_html_shows_study_numbers_not_ledger_ids(two_studies) -> None:
    entries, cites = ledger_entries(two_studies), citations(two_studies)
    one = cited_html(two_studies.prose.summary[1], entries, cites)
    many = cited_html(two_studies.prose.summary[2], entries, cites)

    assert f'[<a href="#{anchor("s1")}">Study 1</a>]' in one
    assert "c1" not in one and "F-flow" not in one
    # one bracket per sentence, each study once, in number order
    assert many.endswith(
        f' [Studies <a href="#{anchor("s1")}">1</a>, <a href="#{anchor("s2")}">2</a>]'
    )


def test_cited_html_escapes_model_text(ui_state) -> None:
    s = Sentence(text='<script>x</script> see [here](javascript:alert(1)) "quoted"', cites=["c1"])
    html = cited_html(s, ledger_entries(ui_state), citations(ui_state))

    assert "<script>" not in html
    assert "](javascript" not in html


def test_sources_show_quote_link_and_tool_result_in_citation_order(ui_state) -> None:
    cards = sources(ui_state)

    assert [(c.key, c.label) for c in cards] == [("res1", "Analysis 1"), ("s1", "Study 1")]
    result, study = cards
    assert "rct_effect" in result.markdown and '"estimate"' in result.markdown
    assert "Ruetzler 2013" in study.markdown
    # the quote links to its page, scrolled to the first quoted passage
    assert "https://pubmed.ncbi.nlm.nih.gov/23921656#:~:text=sore%20throat" in study.markdown


def test_sources_group_a_studys_claims_on_one_card(two_studies) -> None:
    cards = sources(two_studies)

    assert [c.label for c in cards] == ["Analysis 1", "Study 1", "Study 2"]
    assert "236 adults" in cards[1].markdown and "Licorice gargle reduces" in cards[1].markdown
    assert "Agarwal 2009" in cards[2].markdown


def test_sources_skip_uncited_and_unknown_ids(ui_state) -> None:
    ui_state.edges = []
    ui_state.prose = ReportDraft(
        summary=[Sentence(text="x", cites=["F-flow", "missing"])], comparison=[], limitations=[]
    )
    assert sources(ui_state) == []


def test_every_verdict_has_a_plain_meaning_and_the_guide_marks_the_picked_one() -> None:
    guide = verdict_guide("published_only")

    assert set(VERDICT_MEANINGS) == set(VERDICT_LABELS)
    for label in VERDICT_LABELS.values():
        assert label in guide
    picked = [line for line in guide.splitlines() if "this run" in line]
    assert len(picked) == 1 and "📚 Published evidence only" in picked[0]


def run_app(runs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    return app(runs_dir, monkeypatch)


def test_app_shows_a_saved_run(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    at = app(tmp_path, monkeypatch, run="20261102-000000-bbbbbb")

    assert not at.exception
    text = " ".join(m.value for m in [*at.markdown, *at.subheader])
    assert "✅ Estimated from data" in text
    assert "17.4" in text and "Ruetzler 2013" in text
    assert "Short answer" in text and "What the research and analysis found" in text
    assert "Plain summary" not in text and "[c1]" not in text and "[res1]" not in text
    assert [m.value for m in at.metric][:3] == ["1", "1", "1"]  # found → deduped → included


def test_no_tabs_and_no_dropdown_the_ask_box_heads_the_page(ui_state, tmp_path, monkeypatch):
    save(ui_state, tmp_path / "runs", "20261102-000000-bbbbbb")
    url = f"sqlite:///{tmp_path / 'board.db'}"
    at = app(tmp_path / "runs", monkeypatch, board_url=url)

    assert not at.exception
    assert not at.tabs and not at.selectbox
    asks = [t.label for t in at.main.text_input if t.key != "search"]
    assert asks == ["Your cause-and-effect question"]
    assert at.main.subheader[0].value.startswith("Ask the Genie")
    assert not at.sidebar.text_input
    assert at.main.expander[0].label == "How is this different from asking a chatbot?"


def test_the_chatbot_note_shows_even_when_asking_is_off(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)

    assert not at.exception
    labels = [e.label for e in at.main.expander]
    assert "How is this different from asking a chatbot?" in labels


def test_app_with_no_saved_runs(tmp_path, monkeypatch) -> None:
    at = run_app(tmp_path, monkeypatch)

    assert not at.exception
    assert any("No saved runs" in m.value for m in at.info)


# --- the gallery: cards grouped into category bands ------------------------------------------


def test_a_card_shows_the_question_verdict_short_answer_and_genie(ui_state) -> None:
    ui_state.genie = "Licorice: not just for candy."
    c = card(ui_state, tag="full trial")

    assert (c.run_id, c.question, c.tag) == (ui_state.run_id, ui_state.question, "full trial")
    assert c.verdict == "✅ Estimated from data"
    assert c.answer == "At 30 min, risk fell by 17.4 percentage points."
    assert c.genie == "Licorice: not just for candy."


def test_a_card_escapes_model_text_and_clips_long_text(ui_state) -> None:
    ui_state.genie = "<b>x</b> " + "word " * 100
    c = card(ui_state)

    assert "<b>" not in c.genie and len(c.genie) <= GENIE_CHARS + 20
    assert c.genie.endswith("…")


def test_a_partial_run_card_says_so(ui_state) -> None:
    ui_state.verdict, ui_state.prose = None, None
    c = card(ui_state)

    assert c.verdict.startswith("Partial run") and c.answer == ""


def _card(question: str, run_id: str = "r") -> Card:
    return Card(run_id, question, "", "✅ Estimated from data", "", "")


def test_bands_group_cards_by_category_in_a_fixed_order_with_unknown_last() -> None:
    cards = [
        _card("Do aliens cause crop circles?"),
        _card("Did the new onboarding flow raise 7-day activation?", "a"),
        _card("Does the full moon cause more births?"),
        _card("Does licorice gargling reduce post-op sore throat?"),
        _card("did the new onboarding flow raise 7-day ACTIVATION", "b"),
    ]

    grouped = bands(cards)

    assert [name for name, _ in grouped] == [*CATEGORIES, OTHER][1:]  # no empty band
    business = dict(grouped)["📈 Business (A/B tests)"]
    assert [c.run_id for c in business] == ["a", "b"]
    assert [c.question for c in dict(grouped)[OTHER]] == ["Do aliens cause crop circles?"]


def test_bands_use_the_given_category_for_questions_off_the_list() -> None:
    cards = [_card("Do aliens cause crop circles?"), _card("Does coffee cause gout?")]
    picked = {"Do aliens cause crop circles?": "🎈 Silly & fun"}

    grouped = bands(cards, lambda q: picked.get(q, OTHER))

    assert [name for name, _ in grouped] == ["🎈 Silly & fun", OTHER]


def test_bands_skip_empty_categories() -> None:
    assert bands([]) == []


def test_a_listed_question_keeps_its_category_without_a_model_call(make_llm) -> None:
    llm, fake = make_llm([])

    assert category("does the FULL moon cause more births", llm) == "🎈 Silly & fun"
    assert not fake.requests


def test_a_new_question_gets_the_category_the_model_picks(make_llm) -> None:
    llm, fake = make_llm(['{"topic": "health"}'])

    assert category("Does coffee cause gout?", llm) == "🩺 Health"
    (request,) = fake.requests
    assert request["messages"][-1]["content"] == "<question>\nDoes coffee cause gout?\n</question>"


@pytest.mark.parametrize("answer", ['{"topic": "politics"}', "not json"])
def test_a_bad_model_answer_files_the_question_under_miscellaneous(make_llm, answer) -> None:
    llm, _ = make_llm([answer, answer])  # the answer and its one repair retry

    assert category("Does coffee cause gout?", llm) == OTHER == "🗂️ Miscellaneous"


def test_without_a_model_a_new_question_is_miscellaneous() -> None:
    assert category("Does coffee cause gout?", None) == OTHER


def test_every_topic_the_model_can_pick_is_a_band() -> None:
    assert set(TOPICS.values()) == {*CATEGORIES, OTHER}


def test_the_gallery_shows_a_card_per_run_in_its_band(ui_state, tmp_path, monkeypatch) -> None:
    _two_runs(ui_state, tmp_path)
    at = app(tmp_path, monkeypatch)

    assert not at.exception
    headings = [s.value for s in at.subheader]
    assert "🩺 Health" in headings and OTHER in headings
    texts = " ".join(m.value for m in at.markdown)
    assert ui_state.question in texts and "Does coffee cause gout?" in texts
    assert "Short answer" not in texts  # no full answer until one is opened
    opens = [b.label for b in at.button if (b.key or "").startswith("open-")]
    assert opens == ["Read full answer →"] * 2


def test_reading_a_card_opens_its_full_answer_and_back_returns(ui_state, tmp_path, monkeypatch):
    run_id = _two_runs(ui_state, tmp_path)
    at = app(tmp_path, monkeypatch)

    at.button(key=f"open-{run_id}").click().run()

    assert not at.exception
    assert at.query_params["run"] == run_id
    assert any("Short answer" in m.value for m in at.markdown)
    assert any(run_id in c.value for c in at.caption)
    assert not at.main.text_input  # the answer page is for reading

    at.button(key="back").click().run()

    assert "run" not in at.query_params
    assert at.button(key=f"open-{run_id}")


def test_arrows_page_through_a_band(ui_state, tmp_path, monkeypatch) -> None:
    for i in range(PER_PAGE + 1):
        other = ui_state.model_copy(update={"question": f"Does thing {i} cause gout?"})
        save(other, tmp_path, f"2026110{i}-000000-aaaaaa")
    at = app(tmp_path, monkeypatch)
    opened = lambda: [b.key for b in at.button if (b.key or "").startswith("open-")]  # noqa: E731

    assert len(opened()) == PER_PAGE
    assert at.button(key=f"prev-{OTHER}").disabled
    at.button(key=f"next-{OTHER}").click().run()

    assert opened() == ["open-20261100-000000-aaaaaa"]  # newest first, so the oldest is last
    assert at.button(key=f"next-{OTHER}").disabled
    at.button(key=f"prev-{OTHER}").click().run()
    assert len(opened()) == PER_PAGE


# --- the board in the sidebar (U13b) and deep links (U14) ----------------------------------


def app(
    runs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    board_url: str = "",
    env: dict[str, str] | None = None,
    **params: str,
) -> AppTest:
    """The app on `runs_dir`, with `BOARD_DB_URL` and `env` set only if given, opened with
    the query `params`."""
    monkeypatch.setenv("CG_RUNS_DIR", str(runs_dir))
    for name in ("BOARD_DB_URL", "NEBIUS_API_KEY", "NEBIUS_BASE_URL", "CG_LIVE_PASSCODE"):
        monkeypatch.delenv(name, raising=False)
    if board_url:
        monkeypatch.setenv("BOARD_DB_URL", board_url)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    at = AppTest.from_file(str(APP), default_timeout=30)
    for name, value in params.items():
        at.query_params[name] = value
    return at.run()


def test_without_a_database_or_passcode_asking_is_off(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    at = app(tmp_path, monkeypatch)

    assert not at.exception
    assert [t.key for t in at.text_input] == ["search"]
    assert any("Asking is off" in i.value for i in at.main.info)
    assert at.button(key="open-20261102-000000-bbbbbb")


def test_the_sidebar_shows_the_board_to_vote_on(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path / "runs", "20261102-000000-bbbbbb")
    url = f"sqlite:///{tmp_path / 'board.db'}"
    db = board_db.connect(url)
    qid = board_db.add_question(db, "Does green tea cause calm?", "asker")
    board_db.set_status(db, qid, "shown")

    at = app(tmp_path / "runs", monkeypatch, board_url=url)

    assert not at.exception
    shown = " ".join(m.value for m in at.sidebar.markdown)
    assert board.plain("Does green tea cause calm?") in shown
    assert at.sidebar.button(key=f"vote-{qid}")
    assert at.button(key="open-20261102-000000-bbbbbb")  # gallery still works


def test_a_broken_database_leaves_the_gallery_working(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    at = app(tmp_path, monkeypatch, board_url="postgresql://nobody@127.0.0.1:1/none")

    assert not at.exception
    assert any("Board unavailable" in w.value for w in at.warning)
    assert at.button(key="open-20261102-000000-bbbbbb")


def _two_runs(ui_state: RunState, runs_dir: Path) -> str:
    """Two saved questions; returns the run id of the second."""
    save(ui_state, runs_dir, "20261101-000000-aaaaaa")
    other = ui_state.model_copy(update={"question": "Does coffee cause gout?"})
    save(other, runs_dir, "20261102-000000-bbbbbb")
    return "20261102-000000-bbbbbb"


def _shown_run(at: AppTest) -> list[str]:
    return [c.value for c in at.caption if c.value.startswith("Run `")]


def test_a_run_link_opens_that_run(ui_state, tmp_path, monkeypatch) -> None:
    run_id = _two_runs(ui_state, tmp_path)

    at = app(tmp_path, monkeypatch, run=run_id)

    assert not at.exception
    assert any("Does coffee cause gout?" in t.value for t in at.header)
    assert run_id in _shown_run(at)[0]


def test_a_question_link_opens_its_saved_run(ui_state, tmp_path, monkeypatch) -> None:
    run_id = _two_runs(ui_state, tmp_path)

    at = app(tmp_path, monkeypatch, q="does coffee cause GOUT")

    assert not at.exception
    assert run_id in _shown_run(at)[0]


@pytest.mark.parametrize("run_id", ["../../etc", "20991231-000000-zzzzzz"])
def test_a_bad_run_link_says_so_and_shows_the_gallery(ui_state, tmp_path, monkeypatch, run_id):
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")

    at = app(tmp_path, monkeypatch, run=run_id)

    assert not at.exception
    assert any("isn't in the gallery" in w.value for w in at.warning)
    assert at.button(key="open-20261102-000000-bbbbbb") and not _shown_run(at)


# --- run background note and logo ----------------------------------------------------------


def test_a_run_with_a_context_note_shows_it(ui_state, tmp_path, monkeypatch):
    path = save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    (path.parent / "context.md").write_text("**Demo with synthetic data.** A made-up A/B test.")

    at = app(tmp_path, monkeypatch, run="20261102-000000-bbbbbb")

    assert not at.exception
    assert [i.value for i in at.info][0].startswith("**Demo with synthetic data.**")
    assert any("Short answer" in m.value for m in at.markdown)


def test_a_run_without_a_context_note_shows_none(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    at = app(tmp_path, monkeypatch, run="20261102-000000-bbbbbb")

    assert not at.exception
    assert not any("Demo with synthetic data" in i.value for i in at.info)


def test_the_genie_lamp_heads_the_main_page(tmp_path, monkeypatch) -> None:
    logo = APP.parent / "assets" / "genie.png"
    assert logo.is_file() and logo.stat().st_size < 200_000

    at = app(tmp_path, monkeypatch)

    assert not at.exception
    (image,) = at.get("image")
    assert image.proto.imgs[0].url.endswith(".png")
    assert at.title[0].value == "Causal Genie"


def test_the_sidebar_tells_a_new_visitor_what_to_do_folded(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)

    assert not at.exception
    (how,) = [e for e in at.sidebar.expander if e.label == "How to use it"]
    assert not how.proto.expanded
    assert "Tab" not in " ".join(m.value for m in how.markdown)


def test_the_vote_banner_heads_the_sidebar(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path / "runs", "20261102-000000-bbbbbb")
    url = f"sqlite:///{tmp_path / 'board.db'}"
    at = app(tmp_path / "runs", monkeypatch, board_url=url)

    assert not at.exception
    first = at.sidebar.markdown[0].value
    assert "Vote on what's next" in first and "up to one top-voted question a day" in first


def test_without_a_board_there_is_no_vote_banner(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)

    assert not any("Vote on what's next" in m.value for m in at.sidebar.markdown)


# --- the gallery lists runs, so two runs of one question can both be shown -----------------


def _labelled(state: RunState, runs_dir: Path, run_id: str, label: str = "") -> Path:
    path = save(state, runs_dir, run_id)
    if label:
        (path.parent / "label.txt").write_text(f"  {label}\n")
    return path


def test_gallery_runs_list_each_labelled_run_of_a_question(ui_state, tmp_path) -> None:
    full = _labelled(ui_state, tmp_path, "20261101-000000-aaaaaa", "full A/B data")
    kept = _labelled(ui_state, tmp_path, "20261102-000000-bbbbbb", "completers only")

    q = ui_state.question
    assert gallery_runs(tmp_path) == {
        f"{q} · full A/B data": (q, full),
        f"{q} · completers only": (q, kept),
    }


def test_gallery_runs_keep_the_latest_unlabelled_run_of_a_question(ui_state, tmp_path) -> None:
    _labelled(ui_state, tmp_path, "20261101-000000-aaaaaa")
    latest = _labelled(ui_state, tmp_path, "20261102-000000-bbbbbb")

    assert gallery_runs(tmp_path) == {ui_state.question: (ui_state.question, latest)}


def test_each_labelled_run_gets_its_own_card(ui_state, tmp_path, monkeypatch) -> None:
    _labelled(ui_state, tmp_path, "20261101-000000-aaaaaa", "full A/B data")
    _labelled(ui_state, tmp_path, "20261102-000000-bbbbbb", "completers only")

    assert [c.tag for c in gallery_cards(tmp_path)] == ["completers only", "full A/B data"]
    at = app(tmp_path, monkeypatch)
    assert at.button(key="open-20261101-000000-aaaaaa")
    assert at.button(key="open-20261102-000000-bbbbbb")


# --- the one ask box, on top of the main page -----------------------------------------------


def _ask_app(tmp_path: Path, monkeypatch, ui_state, board: bool = True, **env: str):
    save(ui_state, tmp_path / "runs", "20261102-000000-bbbbbb")
    url = f"sqlite:///{tmp_path / 'board.db'}" if board else ""
    at = app(tmp_path / "runs", monkeypatch, board_url=url, env=env)
    return at, (board_db.connect(url) if url else None)


def test_the_ask_box_posts_to_the_board(ui_state, tmp_path, monkeypatch) -> None:
    at, _ = _ask_app(tmp_path, monkeypatch, ui_state)
    ask = at.main

    assert [t.label for t in ask.text_input if t.key != "search"] == [
        "Your cause-and-effect question"
    ]
    assert "Post to the board" in [b.label for b in ask.button]
    assert "Run privately now" not in [b.label for b in ask.button]  # no passcode set


def test_asking_a_gallery_question_points_to_its_answer(ui_state, tmp_path, monkeypatch) -> None:
    # No run starts and no passcode is checked: the gallery already has the answer.
    at, _ = _ask_app(tmp_path, monkeypatch, ui_state, board=False, CG_LIVE_PASSCODE="sesame")

    at.text_input[0].input(ui_state.question.upper())
    next(b for b in at.button if b.label == "Run privately now").click().run()

    assert not at.exception
    (note,) = [i.value for i in at.info if "already answered" in i.value]
    assert "?q=" in note and not at.error


def test_a_passcode_run_uses_the_same_box(ui_state, tmp_path, monkeypatch) -> None:
    at, _ = _ask_app(tmp_path, monkeypatch, ui_state, board=False, CG_LIVE_PASSCODE="sesame")

    asks = [t.label for t in at.text_input if t.key != "search"]
    assert asks == ["Your cause-and-effect question", "Passcode"]
    assert "Post to the board" not in [b.label for b in at.button]  # no board database

    at.text_input[0].input("Does coffee cause gout?")
    at.text_input[1].input("wrong")
    next(b for b in at.button if b.label == "Run privately now").click().run()

    assert not at.exception
    assert any("Wrong passcode" in e.value for e in at.error)


def test_verdict_list_shows_each_name_with_its_meaning_folded() -> None:
    md = verdict_list()

    assert md.count("<details>") == len(VERDICT_LABELS)
    for verdict, label in VERDICT_LABELS.items():
        assert f"<summary>{label}</summary>" in md
        assert VERDICT_MEANINGS[verdict] in md
    assert "<details open" not in md


def test_the_sidebar_lists_every_verdict_unfolded(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)

    assert not at.exception
    assert any(m.value == verdict_list() for m in at.sidebar.markdown)
    assert not [e for e in at.sidebar.expander if "verdict" in e.label.lower()]


def test_the_answer_page_has_a_way_back_at_the_bottom_too(ui_state, tmp_path, monkeypatch):
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    at = app(tmp_path, monkeypatch, run="20261102-000000-bbbbbb")

    at.button(key="back-bottom").click().run()

    assert "run" not in at.query_params and at.button(key="open-20261102-000000-bbbbbb")


# --- searching the answered questions ------------------------------------------------------


def test_search_matches_every_word_in_the_question_or_tag_in_any_case() -> None:
    cards = [
        Card("a", "Does coffee cause gout?", "", "", "", ""),
        Card("b", "Did the new onboarding flow raise activation?", "full A/B data", "", "", ""),
        Card("c", "Does coffee improve focus?", "", "", "", ""),
    ]

    assert [c.run_id for c in matching(cards, "  COFFEE ")] == ["a", "c"]
    assert [c.run_id for c in matching(cards, "gout coffee")] == ["a"]
    assert [c.run_id for c in matching(cards, "a/b onboarding")] == ["b"]
    assert matching(cards, "") == cards
    assert matching(cards, "tea") == []


def test_the_search_box_narrows_the_gallery(ui_state, tmp_path, monkeypatch) -> None:
    run_id = _two_runs(ui_state, tmp_path)
    at = app(tmp_path, monkeypatch)

    at.text_input(key="search").input("gout").run()

    opened = [b.key for b in at.button if (b.key or "").startswith("open-")]
    assert opened == [f"open-{run_id}"]

    at.text_input(key="search").input("unicorns").run()

    assert not [b for b in at.button if (b.key or "").startswith("open-")]
    assert any("No answered question matches" in i.value for i in at.info)


def test_the_report_downloads_as_a_pdf(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    at = app(tmp_path, monkeypatch, run="20261102-000000-bbbbbb")

    assert not at.exception
    (download,) = at.get("download_button")
    assert download.proto.label == "Download report (PDF)"


# --- filtering the answered questions by verdict ---------------------------------------------


def test_verdict_filter_keeps_the_picked_verdicts_and_none_picked_keeps_all() -> None:
    cards = [
        Card("a", "A?", "", "✅ Estimated from data", "", ""),
        Card("b", "B?", "", "📚 Published evidence only", "", ""),
        Card("c", "C?", "", "🔀 Explained by a common cause", "", ""),
    ]

    assert [c.run_id for c in with_verdicts(cards, ["✅ Estimated from data"])] == ["a"]
    two = ["🔀 Explained by a common cause", "✅ Estimated from data"]
    assert [c.run_id for c in with_verdicts(cards, two)] == ["a", "c"]
    assert with_verdicts(cards, []) == cards


def test_verdict_choices_are_the_gallerys_verdicts_in_the_usual_order() -> None:
    cards = [
        Card("a", "A?", "", "🔀 Explained by a common cause", "", ""),
        Card("b", "B?", "", "✅ Estimated from data", "", ""),
        Card("c", "C?", "", "Partial run: no verdict reached", "", ""),
        Card("d", "D?", "", "✅ Estimated from data", "", ""),
    ]

    assert verdict_choices(cards) == [
        "✅ Estimated from data",
        "🔀 Explained by a common cause",
        "Partial run: no verdict reached",
    ]


def test_the_verdict_filter_narrows_the_gallery(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path, "20261101-000000-aaaaaa")
    other = ui_state.model_copy(
        update={"question": "Does coffee cause gout?", "verdict": "published_only"}
    )
    save(other, tmp_path, "20261102-000000-bbbbbb")
    at = app(tmp_path, monkeypatch)
    picker = at.multiselect(key="verdicts")

    assert picker.options == ["✅ Estimated from data", "📚 Published evidence only"]
    picker.select("📚 Published evidence only").run()

    opened = [b.key for b in at.button if (b.key or "").startswith("open-")]
    assert opened == ["open-20261102-000000-bbbbbb"]


def test_the_header_says_what_the_genie_does_and_why_to_trust_it(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)

    assert not at.exception
    assert any(m.value.startswith("**Does X really cause Y?**") for m in at.markdown)
    assert any("traces to a tested tool" in c.value for c in at.caption)


# --- when each question was answered -----------------------------------------------------------


@pytest.mark.parametrize(
    ("run_id", "shown"),
    [("20261117-172931-b755cb", "Nov 17, 2026"), ("20260105-000000-aaaaaa", "Jan 5, 2026")],
)
def test_the_answer_date_comes_from_the_run_id(run_id, shown) -> None:
    assert answered_on(run_id) == shown


@pytest.mark.parametrize("run_id", ["live-run", "20261399-000000-aaaaaa", ""])
def test_a_run_id_without_a_date_shows_none(run_id) -> None:
    assert answered_on(run_id) == ""


def test_a_rerun_replaces_the_card_and_its_date(ui_state, tmp_path) -> None:
    save(ui_state, tmp_path, "20261101-000000-aaaaaa")
    save(ui_state, tmp_path, "20270301-000000-bbbbbb")  # new evidence, same question

    (only,) = gallery_cards(tmp_path)
    assert (only.run_id, only.answered) == ("20270301-000000-bbbbbb", "Mar 1, 2027")


def test_cards_and_the_answer_page_show_when_it_was_answered(ui_state, tmp_path, monkeypatch):
    save(ui_state, tmp_path, "20261117-172931-b755cb")
    at = app(tmp_path, monkeypatch)

    assert any("Answered Nov 17, 2026" in c.value for c in at.caption)

    at.button(key="open-20261117-172931-b755cb").click().run()

    (note,) = [c.value for c in at.caption if c.value.startswith("Answered")]
    assert "Nov 17, 2026" in note and "re-run" in note


def test_the_pdf_is_dated(ui_state, tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr("causal_genie.ui.pdf.dated", lambda md, day: calls.append((md, day)) or md)
    save(ui_state, tmp_path, "20261117-172931-b755cb")
    st.cache_data.clear()  # the app keeps each report's PDF

    at = app(tmp_path, monkeypatch, run="20261117-172931-b755cb")

    assert not at.exception
    assert calls and calls[-1][1] == "Nov 17, 2026"


def test_the_most_recently_answered_question_comes_first(ui_state, tmp_path, monkeypatch) -> None:
    save(
        ui_state.model_copy(update={"question": "Does tea cause gout?"}),
        tmp_path,
        "20261101-000000-aaaaaa",
    )
    save(
        ui_state.model_copy(update={"question": "Does beer cause gout?"}),
        tmp_path,
        "20261103-000000-cccccc",
    )
    save(
        ui_state.model_copy(update={"question": "Does wine cause gout?"}),
        tmp_path,
        "20261102-000000-bbbbbb",
    )

    assert [c.run_id[:8] for c in gallery_cards(tmp_path)] == ["20261103", "20261102", "20261101"]
    at = app(tmp_path, monkeypatch)
    opened = [b.key for b in at.button if (b.key or "").startswith("open-")]
    assert opened == [
        "open-20261103-000000-cccccc",
        "open-20261102-000000-bbbbbb",
        "open-20261101-000000-aaaaaa",
    ]


def test_the_passcode_option_gives_the_real_run_time(ui_state, tmp_path, monkeypatch) -> None:
    at, _ = _ask_app(tmp_path, monkeypatch, ui_state, board=False, CG_LIVE_PASSCODE="sesame")

    (box,) = [e for e in at.expander if e.label.startswith("Have a passcode?")]
    assert "about 5 to 10 minutes" in box.label  # a live run on the site took 10 min 42 s
