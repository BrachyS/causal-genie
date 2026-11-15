"""Streamlit UI (T11.1): saved-run lookup, cited prose with clickable numbers, source cards,
and the app itself on a saved run (Streamlit's AppTest, no browser, no network)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from causal_genie.agent.state import ReportDraft, Result, RunState, Sentence
from causal_genie.agent.steps.report import VERDICT_LABELS, ledger_entries, write_report
from causal_genie.tools.stats import risk_difference
from causal_genie.ui.view import (
    VERDICT_MEANINGS,
    anchor,
    citations,
    cited_html,
    saved_runs,
    sources,
    verdict_guide,
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
    monkeypatch.setenv("CG_RUNS_DIR", str(runs_dir))
    return AppTest.from_file(str(APP), default_timeout=30).run()


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


def test_app_says_when_a_question_has_no_saved_run(ui_state, tmp_path, monkeypatch) -> None:
    save(ui_state, tmp_path, "20261102-000000-bbbbbb")
    at = run_app(tmp_path, monkeypatch)
    at.text_input[0].input("Does coffee cause gout?").run()

    assert not at.exception
    assert any("No saved run" in m.value for m in at.info)


def test_app_with_no_saved_runs(tmp_path, monkeypatch) -> None:
    at = run_app(tmp_path, monkeypatch)

    assert not at.exception
    assert any("No saved runs" in m.value for m in at.info)
