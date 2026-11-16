"""Streamlit UI (T11.1): saved-run lookup, cited prose with clickable numbers, source cards,
and the app itself on a saved run (Streamlit's AppTest, no browser, no network)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from causal_genie.agent.state import ReportDraft, Result, RunState, Sentence
from causal_genie.agent.steps.report import VERDICT_LABELS, ledger_entries, write_report
from causal_genie.tools.stats import risk_difference
from causal_genie.ui import board, board_db
from causal_genie.ui.view import (
    VERDICT_MEANINGS,
    anchor,
    citations,
    cited_html,
    gallery_runs,
    saved_runs,
    sources,
    verdict_guide,
)

APP = Path(__file__).parents[1] / "causal_genie" / "ui" / "app.py"
TABS = ("📚 Gallery", "🙋 Ask & vote")


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
    at = run_app(tmp_path, monkeypatch)

    assert not at.exception
    text = " ".join(m.value for m in [*at.markdown, *at.subheader])
    assert "✅ Estimated from data" in text
    assert "17.4" in text and "Ruetzler 2013" in text
    assert "Short answer" in text and "What the research and analysis found" in text
    assert "Plain summary" not in text and "[c1]" not in text and "[res1]" not in text
    assert [m.value for m in at.metric][:3] == ["1", "1", "1"]  # found → deduped → included


def test_the_gallery_is_read_only_and_typing_happens_in_one_place(
    ui_state, tmp_path, monkeypatch
) -> None:
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    at = run_app(tmp_path, monkeypatch)

    gallery, ask = at.tabs
    assert (gallery.label, ask.label) == TABS
    assert not gallery.text_input and gallery.selectbox[0].value == ui_state.question
    assert not at.sidebar.radio


def test_app_with_no_saved_runs(tmp_path, monkeypatch) -> None:
    at = run_app(tmp_path, monkeypatch)

    assert not at.exception
    assert any("No saved runs" in m.value for m in at.info)


# --- board tab (U13b) and deep links (U14) -----------------------------------------------


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
    assert not at.tabs[1].text_input
    assert any("Asking is off" in i.value for i in at.tabs[1].info)
    assert any("Short answer" in m.value for m in at.tabs[0].markdown)


def test_board_tab_shows_the_questions_on_the_board(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path / "runs", "20261102-000000-bbbbbb")
    url = f"sqlite:///{tmp_path / 'board.db'}"
    db = board_db.connect(url)
    qid = board_db.add_question(db, "Does green tea cause calm?", "asker")
    board_db.set_status(db, qid, "shown")

    at = app(tmp_path / "runs", monkeypatch, board_url=url)

    assert not at.exception
    assert tuple(t.label for t in at.tabs) == TABS
    shown = " ".join(m.value for m in at.tabs[1].markdown)
    assert board.plain("Does green tea cause calm?") in shown
    assert any("Short answer" in m.value for m in at.tabs[0].markdown)  # gallery still works


def test_a_broken_database_leaves_the_gallery_working(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    at = app(tmp_path, monkeypatch, board_url="postgresql://nobody@127.0.0.1:1/none")

    assert not at.exception
    assert any("Board unavailable" in w.value for w in at.warning)
    assert any("Short answer" in m.value for m in at.markdown)


def _two_runs(ui_state: RunState, runs_dir: Path) -> str:
    """Two saved questions; returns the run id of the second (not the default pick)."""
    save(ui_state, runs_dir, "20261101-000000-aaaaaa")
    other = ui_state.model_copy(update={"question": "Does coffee cause gout?"})
    save(other, runs_dir, "20261102-000000-bbbbbb")
    return "20261102-000000-bbbbbb"


def test_a_run_link_opens_that_run(ui_state, tmp_path, monkeypatch) -> None:
    run_id = _two_runs(ui_state, tmp_path)

    at = app(tmp_path, monkeypatch, run=run_id)

    assert not at.exception
    assert at.selectbox[0].value == "Does coffee cause gout?"
    assert any(run_id in c.value for c in at.caption)


def test_a_question_link_opens_its_saved_run(ui_state, tmp_path, monkeypatch) -> None:
    _two_runs(ui_state, tmp_path)

    at = app(tmp_path, monkeypatch, q="does coffee cause GOUT")

    assert not at.exception
    assert at.selectbox[0].value == "Does coffee cause gout?"


def test_after_a_link_the_gallery_picker_still_works(ui_state, tmp_path, monkeypatch) -> None:
    run_id = _two_runs(ui_state, tmp_path)
    at = app(tmp_path, monkeypatch, run=run_id)

    at.selectbox[0].set_value(ui_state.question).run()

    assert at.selectbox[0].value == ui_state.question
    assert any("20261101-000000-aaaaaa" in c.value for c in at.caption)


@pytest.mark.parametrize("run_id", ["../../etc", "20991231-000000-zzzzzz"])
def test_a_bad_run_link_says_so_and_shows_the_gallery(ui_state, tmp_path, monkeypatch, run_id):
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")

    at = app(tmp_path, monkeypatch, run=run_id)

    assert not at.exception
    assert any("isn't in the gallery" in w.value for w in at.warning)
    assert at.selectbox[0].value == ui_state.question


# --- run background note and logo ----------------------------------------------------------


def test_a_run_with_a_context_note_shows_it(ui_state, tmp_path, monkeypatch):
    path = save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    (path.parent / "context.md").write_text("**Demo with synthetic data.** A made-up A/B test.")

    at = app(tmp_path, monkeypatch)

    assert not at.exception
    assert [i.value for i in at.info][0].startswith("**Demo with synthetic data.**")
    assert any("Short answer" in m.value for m in at.markdown)


def test_a_run_without_a_context_note_shows_none(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    at = app(tmp_path, monkeypatch)

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


def test_the_picker_shows_the_exact_run_picked(ui_state, tmp_path, monkeypatch) -> None:
    _labelled(ui_state, tmp_path, "20261101-000000-aaaaaa", "full A/B data")
    _labelled(ui_state, tmp_path, "20261102-000000-bbbbbb", "completers only")
    at = app(tmp_path, monkeypatch)
    q = ui_state.question

    assert at.selectbox[0].options == [f"{q} · full A/B data", f"{q} · completers only"]
    assert any("20261101-000000-aaaaaa" in c.value for c in at.caption)

    at.selectbox[0].set_value(f"{q} · completers only").run()

    assert any("20261102-000000-bbbbbb" in c.value for c in at.caption)


def test_a_run_link_picks_its_labelled_run(ui_state, tmp_path, monkeypatch) -> None:
    _labelled(ui_state, tmp_path, "20261101-000000-aaaaaa", "full A/B data")
    _labelled(ui_state, tmp_path, "20261102-000000-bbbbbb", "completers only")

    at = app(tmp_path, monkeypatch, run="20261101-000000-aaaaaa")

    assert at.selectbox[0].value == f"{ui_state.question} · full A/B data"
    assert any("20261101-000000-aaaaaa" in c.value for c in at.caption)


# --- the one ask box (Ask & vote tab) --------------------------------------------------------


def _ask_app(tmp_path: Path, monkeypatch, ui_state, board: bool = True, **env: str):
    save(ui_state, tmp_path / "runs", "20261102-000000-bbbbbb")
    url = f"sqlite:///{tmp_path / 'board.db'}" if board else ""
    at = app(tmp_path / "runs", monkeypatch, board_url=url, env=env)
    return at, (board_db.connect(url) if url else None)


def test_the_ask_box_posts_to_the_board(ui_state, tmp_path, monkeypatch) -> None:
    at, _ = _ask_app(tmp_path, monkeypatch, ui_state)
    ask = at.tabs[1]

    assert [t.label for t in ask.text_input] == ["Your cause-and-effect question"]
    assert "Post to the board" in [b.label for b in ask.button]
    assert "Run privately now" not in [b.label for b in ask.button]  # no passcode set


def test_asking_a_gallery_question_points_to_its_answer(ui_state, tmp_path, monkeypatch) -> None:
    # No run starts and no passcode is checked: the gallery already has the answer.
    at, _ = _ask_app(tmp_path, monkeypatch, ui_state, board=False, CG_LIVE_PASSCODE="sesame")
    ask = at.tabs[1]

    ask.text_input[0].input(ui_state.question.upper())
    next(b for b in ask.button if b.label == "Run privately now").click().run()

    assert not at.exception
    (note,) = [i.value for i in at.info if "already answered" in i.value]
    assert "?q=" in note and not at.error


def test_a_passcode_run_uses_the_same_box(ui_state, tmp_path, monkeypatch) -> None:
    at, _ = _ask_app(tmp_path, monkeypatch, ui_state, board=False, CG_LIVE_PASSCODE="sesame")
    ask = at.tabs[1]

    assert [t.label for t in ask.text_input] == ["Your cause-and-effect question", "Passcode"]
    assert "Post to the board" not in [b.label for b in ask.button]  # no board database

    ask.text_input[0].input("Does coffee cause gout?")
    ask.text_input[1].input("wrong")
    next(b for b in ask.button if b.label == "Run privately now").click().run()

    assert not at.exception
    assert any("Wrong passcode" in e.value for e in at.error)
