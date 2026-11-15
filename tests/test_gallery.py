"""Gallery export (T11.2): a saved run copied for the public demo without raw web text."""

import json

from causal_genie.agent.state import ExtractCall, RunState
from causal_genie.ui.gallery import add_genie, export, main
from causal_genie.ui.view import saved_runs


def test_export_keeps_what_the_app_shows_and_drops_raw_web_text(sample_state, tmp_path) -> None:
    sample_state.extracts = [
        ExtractCall(url="https://x.org/p", content="full page text", error=None, latency_s=1.0)
    ]
    sample_state.report = "# report"
    run_dir = sample_state.save(tmp_path / "runs").parent

    path = export(run_dir, tmp_path / "gallery")

    copy = RunState.load(path)
    assert path == tmp_path / "gallery" / sample_state.run_id / "state.json"
    assert copy.report == "# report"
    assert copy.claims == sample_state.claims and copy.studies == sample_state.studies
    assert copy.searches[0].query == sample_state.searches[0].query
    assert all(h.content == "" for s in copy.searches for h in s.hits)
    assert copy.extracts[0].content is None and copy.extracts[0].url == "https://x.org/p"
    assert all(c.user_content == "" and c.response == "" for c in copy.llm_calls)
    assert copy.tokens == sample_state.tokens
    assert "full page text" not in path.read_text()


def test_cli_exports_runs_by_id_and_the_app_lists_them(sample_state, tmp_path) -> None:
    sample_state.report = "# report"
    sample_state.save(tmp_path / "runs")

    main([sample_state.run_id, "--runs", str(tmp_path / "runs"), "--to", str(tmp_path / "g")])

    assert saved_runs(tmp_path / "g") == {
        sample_state.question: tmp_path / "g" / sample_state.run_id / "state.json"
    }


def gate_output(sensitive: bool = False) -> str:
    return json.dumps(
        {
            "sensitive": sensitive,
            "whimsical": False,
            "aside": "The gargle kept its cool.",
            "next_step": "Read the studies listed above.",
        }
    )


def test_add_genie_gates_a_saved_run_and_records_the_call(sample_state, tmp_path, make_llm) -> None:
    sample_state.verdict = "estimated"
    sample_state.report = "# old report"
    run_dir = sample_state.save(tmp_path).parent
    llm, _ = make_llm([gate_output()])

    state = add_genie(run_dir, llm)

    saved = RunState.load(run_dir / "state.json")
    assert saved.genie == state.genie == "The gargle kept its cool. Read the studies listed above."
    assert saved.llm_calls[-1].prompt_name == "genie_gate"  # replay turns Genie on from this
    assert saved.report is not None and "🧞 Genie's aside" in saved.report
    assert (run_dir / "report.md").read_text() == saved.report


def test_add_genie_leaves_a_gated_run_alone(sample_state, tmp_path, make_llm) -> None:
    sample_state.verdict = "estimated"
    sample_state.llm_calls[0].prompt_name = "genie_gate"
    run_dir = sample_state.save(tmp_path).parent
    llm, _ = make_llm([])

    add_genie(run_dir, llm)

    assert llm.calls == []
