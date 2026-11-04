import pytest

from causal_genie.agent.state import Download, Result, Route, RunState
from causal_genie.agent.steps.report import write_report
from causal_genie.tools.stats import risk_difference


@pytest.fixture
def estimated_state(sample_state: RunState) -> RunState:
    """The sample run after a risk-difference estimate on downloaded data."""
    outputs = risk_difference(22, 117, 42, 116) | {
        "rows_excluded_missing": 2,
        "treatment_column": "treat",
        "outcome_column": "pacu30min_throatPain",
        "cutoff": 0,
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
    assert "Outcome in the data: `pacu30min_throatPain` > 0" in report
    assert "Treatment in the data: `treat`" in report
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
        "treatment_column": "arm",
        "outcome_column": "score",
        "cutoff": None,
    }

    report = write_report(estimated_state)

    assert "mean 5.25 (n=40)" in report and "mean 4.00 (n=55)" in report
    assert "Mean difference: 1.25 (95% CI 0.50 to 2.00)" in report
    assert "missing values" not in report
    assert "Outcome in the data: `score`\n" in report


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
