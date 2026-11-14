import json

import pytest

from causal_genie.agent.certainty import certainty
from causal_genie.agent.state import (
    Adjustment,
    DagEdge,
    DesignCheck,
    Download,
    Query,
    ReportDraft,
    ResearchPlan,
    ResearchRound,
    Result,
    Route,
    RunState,
    Sentence,
)
from causal_genie.agent.steps.report import (
    draft_report,
    free_numbers,
    ledger_entries,
    sentence_problem,
    write_report,
)
from causal_genie.tools.stats import risk_difference


@pytest.fixture
def estimated_state(sample_state: RunState) -> RunState:
    """The sample run after a risk-difference estimate on downloaded data."""
    outputs = risk_difference(22, 117, 42, 116) | {
        "rows_excluded_missing": 2,
        "mapped_from": {
            "licorice_gargle": "treat",
            "sore_throat": "pacu30min_throatPain > 0",
        },
    }
    sample_state.results = [
        Result(id="res1", tool="rct_effect", tool_version="0.1.0", inputs_hash="h", outputs=outputs)
    ]
    sample_state.downloads = [
        Download(
            url="https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/licorice_gargle.csv",
            path="cache/downloads/abc.csv",
            sha256="2cc1078d",
            bytes=11208,
            retrieved_at="2026-11-03T16:00:00+00:00",
            doi=None,
            license="MIT",
        )
    ]
    sample_state.verdict = "estimated"
    return sample_state


def test_estimated_report_shows_verdict_estimate_data_and_sources(estimated_state) -> None:
    report = write_report(estimated_state)

    assert "✅ Estimated from data" in report
    assert "18.8% (22/117)" in report and "36.2% (42/116)" in report
    assert "-17.4 percentage points" in report
    assert "95% CI -28.3 to -5.9" in report
    assert "2 rows with missing values were excluded" in report
    assert (
        "In the data: `licorice_gargle` ← `treat`; `sore_throat` ← `pacu30min_throatPain > 0`"
        in report
    )
    assert "licorice_gargle.csv" in report and "MIT" in report and "2cc1078d" in report
    assert "https://pubmed.ncbi.nlm.nih.gov/23921656" in report
    assert "not medical advice" in report
    assert "r1" in report


def test_mean_difference_report(estimated_state) -> None:
    estimated_state.results[0].outputs = {
        "scale": "mean_difference",
        "estimate": 1.25,
        "ci_low": 0.5,
        "ci_high": 2.0,
        "level": 0.95,
        "method": "Welch",
        "n_treated": 40,
        "mean_treated": 5.25,
        "n_control": 55,
        "mean_control": 4.0,
        "rows_excluded_missing": 0,
        "mapped_from": {"licorice_gargle": "arm", "sore_throat": "score"},
    }

    report = write_report(estimated_state)

    assert "mean 5.25 (n=40)" in report and "mean 4.00 (n=55)" in report
    assert "Mean difference: 1.25 (95% CI 0.50 to 2.00)" in report
    assert "missing values" not in report
    assert "`sore_throat` ← `score`" in report


def test_report_without_estimate_says_so(sample_state: RunState) -> None:
    sample_state.verdict = "published_only"

    report = write_report(sample_state)

    assert "📚 Published evidence only" in report
    assert "No estimate" in report


def test_not_causal_report_offers_rewrite(sample_state: RunState) -> None:
    sample_state.query = None
    sample_state.route = Route(
        kind="not_causal",
        run_id=None,
        reason="asks for a count",
        causal_rewrite="Does the full moon cause more births?",
    )
    sample_state.verdict = "not_causal"

    report = write_report(sample_state)

    assert "🚫 Not causal" in report
    assert "Does the full moon cause more births?" in report


def test_evidence_section_shows_certainty_flow_and_checked_effects(sample_state) -> None:
    sample_state.verdict = "published_only"
    sample_state.certainty = certainty(sample_state.studies)
    sample_state.studies.append(
        sample_state.studies[0].model_copy(
            update={"id": "s2", "included": False, "exclusion_reason": "off_topic"}
        )
    )

    report = write_report(sample_state)

    assert "**Certainty:** high (starts high" in report
    assert "2 sources found → 2 studies after dedupe → 1 included" in report
    assert "excluded: off_topic 1" in report
    study = sample_state.studies[0]
    assert f"[{study.title}]({study.url})" in report
    assert "sore throat 30 min: 24% vs 61%" in report  # backed by documented claim c1


def test_unverified_effect_is_not_reported(sample_state) -> None:
    sample_state.verdict = "published_only"
    sample_state.claims[0].status = "unresolved"

    report = write_report(sample_state)

    assert "24% vs 61%" not in report
    assert "effect not verified" in report


def test_insufficient_report_lists_what_was_searched(sample_state) -> None:
    sample_state.verdict = "insufficient"
    sample_state.studies, sample_state.claims = [], []
    query = Query(
        text="licorice gargle sore throat", tool="search", group="primary_studies", why="w"
    )
    plan = ResearchPlan(queries=[query], done=True, done_reason="nothing")
    sample_state.rounds = [ResearchRound(number=1, plan=plan, planned_by="model")]

    report = write_report(sample_state)

    assert "❓" in report
    assert "No included studies" in report
    assert "- licorice gargle sore throat" in report


def test_observational_report_shows_adjusted_and_naive_with_precision_apart(
    estimated_state: RunState,
) -> None:
    estimated_state.results[0] = Result(
        id="res2",
        tool="backdoor_effect",
        tool_version="0.1.0",
        inputs_hash="h",
        outputs={
            "scale": "mean_difference",
            "estimate": -12.04,
            "ci_low": -108.1,
            "ci_high": 84.0,
            "level": 0.95,
            "method": "OLS, HC1 robust SEs",
            "n": 7305,
            "n_treated": 247,
            "adjusted_for": [],
            "precision": ["holiday", "weekday"],
            "naive_estimate": 30.5,
            "naive_ci_low": -310.2,
            "naive_ci_high": 371.2,
            "rows": 7305,
            "rows_excluded_missing": 0,
            "mapped_from": {"full_moon": "full_moon(date)", "births": "births"},
        },
    )

    report = write_report(estimated_state)

    assert "Adjusted mean difference: -12.04 (95% CI -108.10 to 84.00)" in report
    assert "Unadjusted: 30.50 (95% CI -310.20 to 371.20)" in report
    assert "Confounders adjusted for: none (no backdoor path in the DAG)" in report
    assert "Also adjusted for, to narrow the CI: holiday, weekday" in report
    assert "7305 rows used (247 exposed)" in report
    assert "`full_moon` ← `full_moon(date)`" in report
    assert "Tool result: `res2`" in report


def test_study_plan_report_names_the_design_and_the_check(sample_state: RunState) -> None:
    sample_state.verdict = "study_plan"
    sample_state.design = DesignCheck(
        design="selection_after_assignment",
        estimable=False,
        key_assumption="arms stay comparable only if every randomized unit is kept",
        estimand="none: comparing the kept rows is biased",
        check="ask for the full randomized roster, including units that left early",
        reasons=["arm sizes 2130 treated vs 1646 control are off the planned 50% allocation"],
    )

    report = write_report(sample_state)

    assert "## Study plan" in report
    assert "selection after assignment" in report
    assert "arm sizes 2130 treated vs 1646 control" in report
    assert "ask for the full randomized roster" in report
    assert "No estimate" not in report


def sentence(text: str, *cites: str) -> dict:
    return {"text": text, "cites": list(cites)}


def draft(summary: list[dict], comparison=(), limitations=()) -> str:
    """Report model output as JSON."""
    return json.dumps(
        {"summary": summary, "comparison": list(comparison), "limitations": list(limitations)}
    )


def test_entries_hold_facts_results_and_only_documented_claims(estimated_state) -> None:
    estimated_state.claims.append(
        estimated_state.claims[0].model_copy(update={"id": "c2", "status": "unresolved"})
    )

    entries = ledger_entries(estimated_state)

    assert "F-verdict" in entries and "✅" in entries["F-verdict"]
    assert "-17.4 percentage points" in entries["res1"]
    assert "c1" in entries and '"sore throat ... was reduced"' in entries["c1"]
    assert "Ruetzler 2013" in entries["c1"]
    assert "c2" not in entries  # not documented: the report may not use it


def test_lint_traces_numbers_to_cited_entries(estimated_state) -> None:
    entries = ledger_entries(estimated_state)
    free = free_numbers(estimated_state)

    def check(text: str, *cites: str) -> str | None:
        return sentence_problem(Sentence(text=text, cites=list(cites)), entries, free)

    assert check("Licorice cut sore throat by 17.4 percentage points.", "res1") is None
    assert check("Licorice gargle reduces post-op sore throat.", "c1") is None
    assert check("The trial ran in 2013.", "c1") is None  # year in the entry header
    assert "17.4" in (check("Licorice cut sore throat by 17.4 points.", "c1") or "")
    assert "cites nothing" in (check("Licorice works.") or "")
    assert "c9" in (check("Licorice works.", "c9") or "")
    assert check("Sore throat was checked in the first 24 h.", "F-verdict") is None  # PECO


def test_draft_that_passes_lint_is_kept_after_one_call(estimated_state, make_llm) -> None:
    llm, _ = make_llm([draft([sentence("Licorice gargle cut sore throat.", "c1", "res1")])])

    prose = draft_report(estimated_state, llm)

    assert [s.text for s in prose.summary] == ["Licorice gargle cut sore throat."]
    assert [c.prompt_name for c in llm.calls] == ["report"]
    assert llm.calls[0].tier == "large"


def test_failing_draft_gets_one_rewrite_naming_the_problems(estimated_state, make_llm) -> None:
    bad = draft([sentence("Licorice halves sore throat, a 50% cut.", "c1")])
    good = draft([sentence("Licorice gargle reduces post-op sore throat.", "c1")])
    llm, _ = make_llm([bad, good])

    prose = draft_report(estimated_state, llm)

    assert [s.text for s in prose.summary] == ["Licorice gargle reduces post-op sore throat."]
    rewrite = llm.calls[1].user_content
    assert "summary[1]" in rewrite and "50" in rewrite and "Licorice halves" in rewrite


def test_sentences_still_failing_after_the_rewrite_are_dropped(estimated_state, make_llm) -> None:
    bad = draft(
        [sentence("Licorice gargle reduces post-op sore throat.", "c1")],
        limitations=[sentence("Only 12 patients smoked.", "c1")],
    )
    llm, _ = make_llm([bad, bad])

    prose = draft_report(estimated_state, llm)

    assert len(llm.calls) == 2  # one rewrite, no more
    assert len(prose.summary) == 1 and prose.limitations == []


def test_summary_is_cut_to_120_words(estimated_state, make_llm) -> None:
    long = "Licorice gargle reduces post-op sore throat " + "and more " * 30 + "."
    llm, _ = make_llm([draft([sentence(long, "c1")] * 3)] * 2)

    prose = draft_report(estimated_state, llm)

    assert len(prose.summary) == 1  # 66 words each: two would pass 120


def test_report_renders_prose_with_citations_and_sources(estimated_state) -> None:
    estimated_state.prose = ReportDraft(
        summary=[Sentence(text="Licorice gargle cut sore throat.", cites=["c1", "res1"])],
        comparison=[Sentence(text="Our estimate matches the trial.", cites=["res1", "c1"])],
        limitations=[Sentence(text="The mapping was not confirmed.", cites=["F-data"])],
    )

    report = write_report(estimated_state)

    assert "## Plain summary\n\nLicorice gargle cut sore throat. [c1, res1]" in report
    assert "## Comparison with published\n\nOur estimate matches the trial." in report
    assert "## Limitations\n\n- The mapping was not confirmed." in report
    assert "[F-data]" not in report  # run facts are shown in their own sections
    assert '- [c1] Ruetzler 2013: "sore throat ... was reduced" — https://pubmed' in report
    sections = [line for line in report.splitlines() if line.startswith("## ")]
    assert sections == [
        "## Plain summary",
        "## Question (PECO)",
        "## Causal graph (DAG)",
        "## Key confounders",
        "## Evidence",
        "## Data used",
        "## Estimate",
        "## Comparison with published",
        "## Limitations",
        "## Sources",
    ]


def test_dag_and_confounder_sections(estimated_state, confounder_claim) -> None:
    estimated_state.claims.append(confounder_claim())
    estimated_state.edges = [
        DagEdge(source="licorice_gargle", target="sore_throat", claim_ids=[], question=True),
        DagEdge(source="smoking", target="licorice_gargle", claim_ids=["s1-adjusted_for"]),
        DagEdge(source="smoking", target="sore_throat", claim_ids=[]),
    ]
    estimated_state.adjustment = Adjustment(adjust_for=[], missing=[], randomized=True)

    report = write_report(estimated_state)

    assert "- licorice_gargle → sore_throat: the question being tested" in report
    assert "- smoking → licorice_gargle: cites s1-adjusted_for" in report
    assert "- smoking → sore_throat: assumed (no source)" in report
    assert "- smoking" in report.split("## Key confounders")[1]
    assert "randomized: assignment balances confounders" in report


def test_dag_section_shows_exogenous_nodes_and_the_edges_dropped(estimated_state) -> None:
    estimated_state.edges = [
        DagEdge(source="licorice_gargle", target="sore_throat", claim_ids=[], question=True),
    ]
    estimated_state.exogenous = ["licorice_gargle"]
    estimated_state.exogenous_dropped = [
        DagEdge(source="smoking", target="licorice_gargle", claim_ids=["s1-adjusted_for"]),
    ]

    dag = write_report(estimated_state).split("## Causal graph (DAG)")[1].split("## ")[0]

    assert (
        "Treated as exogenous (the model's judgment that nothing in the graph can cause it; "
        "no source): licorice_gargle" in dag
    )
    assert "Arrows into them were dropped: smoking → licorice_gargle" in dag


def test_dag_section_says_nothing_about_exogenous_nodes_when_there_are_none(
    estimated_state,
) -> None:
    estimated_state.edges = [
        DagEdge(source="licorice_gargle", target="sore_throat", claim_ids=[], question=True),
    ]

    assert "exogenous" not in write_report(estimated_state)


def test_partial_report_names_the_failed_step(sample_state: RunState) -> None:
    sample_state.step = "research"
    sample_state.failure = "APIConnectionError: Connection error."

    report = write_report(sample_state)

    assert "Partial report" in report
    assert "`research` step failed" in report and "APIConnectionError" in report
    assert "not a finding that no evidence exists" in report
    assert "## Question (PECO)" in report  # what ran before the failure is still shown


def test_genie_aside_is_shown_when_set(sample_state: RunState) -> None:
    sample_state.verdict = "published_only"
    sample_state.genie = "The moon keeps its secrets. Next: read the studies above."

    report = write_report(sample_state)

    assert "## 🧞 Genie's aside\n\nThe moon keeps its secrets." in report


def backdoor_result(adjusted_for: list[str], precision: list[str]) -> Result:
    """A D0-like observational result (live run 20261114-154259-6bb1d2's numbers)."""
    return Result(
        id="res3",
        tool="backdoor_effect",
        tool_version="0.1.0",
        inputs_hash="h",
        outputs={
            "scale": "mean_difference",
            "estimate": -10.46,
            "ci_low": -342.96,
            "ci_high": 322.03,
            "level": 0.95,
            "method": "OLS, HC1 robust SEs",
            "n": 3652,
            "n_treated": 123,
            "adjusted_for": adjusted_for,
            "precision": precision,
            "naive_estimate": -10.46,
            "naive_ci_low": -342.96,
            "naive_ci_high": 322.03,
            "rows": 3652,
            "rows_excluded_missing": 0,
            "mapped_from": {"full_moon": "full_moon(date)", "births": "births"},
        },
    )


def test_estimate_with_nothing_adjusted_for_is_called_unadjusted(estimated_state) -> None:
    estimated_state.results[0] = backdoor_result([], [])

    report = write_report(estimated_state)
    entry = ledger_entries(estimated_state)["res3"]

    line = "Unadjusted mean difference: -10.46 (95% CI -342.96 to 322.03), OLS, HC1 robust SEs."
    assert line in report and line in entry
    assert "Adjusted" not in report and "Adjusted" not in entry
    assert "Unadjusted: " not in report  # no second, identical estimate line
    assert "Nothing adjusted for: no backdoor path in the DAG" in report


def test_estimand_shows_the_scale_computed_not_the_one_framed(estimated_state) -> None:
    assert estimated_state.query is not None
    estimated_state.query.effect_scale = "risk_difference"  # as framed in live D0
    estimated_state.results[0] = backdoor_result([], ["day_of_week"])

    report = write_report(estimated_state)

    assert (
        "- **Estimand:** ATE, mean difference (framed as risk difference; the outcome in the "
        "data is not 0/1), deviations handled" in report
    )
    assert "risk difference," not in report


def test_estimand_without_an_estimate_shows_the_framed_scale(sample_state: RunState) -> None:
    assert sample_state.query is not None
    sample_state.verdict = "published_only"

    report = write_report(sample_state)

    scale = sample_state.query.effect_scale.replace("_", " ")
    assert f"- **Estimand:** {sample_state.query.target}, {scale}, deviations" in report


def test_entries_leave_out_study_details_the_report_already_shows(estimated_state) -> None:
    """Design, n and population claims are rendered by the harness (Evidence, Sources); sending
    them to the report model doubled its input in live D0 (T10.9 step 4)."""
    c1 = estimated_state.claims[0]
    estimated_state.claims += [
        c1.model_copy(update={"id": "c-design", "kind": "study_detail"}),
        c1.model_copy(update={"id": "c-null", "kind": "null_result"}),
        c1.model_copy(update={"id": "c-conf", "kind": "confounder"}),
    ]

    entries = ledger_entries(estimated_state)

    assert "c-design" not in entries
    assert {"c1", "c-null", "c-conf"} <= set(entries)
