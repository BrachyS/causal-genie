import pytest

from causal_genie.agent.certainty import certainty
from causal_genie.agent.state import (
    DesignCheck,
    Download,
    Query,
    ResearchPlan,
    ResearchRound,
    Result,
    Route,
    RunState,
)
from causal_genie.agent.steps.report import write_report
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
