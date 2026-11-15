from collections.abc import Callable

import pytest

from causal_genie.agent.certainty import certainty, evidence_flow
from causal_genie.agent.state import Study


@pytest.fixture
def make_study(sample_state) -> Callable[..., Study]:
    """An included, direct study like the sample trial, with the given fields changed."""
    base = sample_state.studies[0]
    count = iter(range(100))

    def make(**changes) -> Study:
        i = next(count)
        return base.model_copy(update={"id": f"s{i}", "url": f"https://s{i}"} | changes)

    return make


def test_rct_matching_the_question_starts_high(make_study) -> None:
    c = certainty([make_study(design="RCT")])

    assert c is not None and c.level == "high"


def test_meta_analysis_of_trials_starts_high(make_study) -> None:
    studies = [make_study(design="meta_analysis", pooled="trials"), make_study(design="cohort")]

    c = certainty(studies)

    assert c is not None and c.level == "high"


@pytest.mark.parametrize("pooled", ["observational", "mixed", "n/a"])
def test_review_not_only_of_trials_counts_as_observational(make_study, pooled: str) -> None:
    c = certainty([make_study(design="systematic_review", pooled=pooled)])

    assert c is not None and c.level == "low"
    assert "observational" in c.reasons[0]


def test_observational_only_starts_low(make_study) -> None:
    c = certainty([make_study(design="cohort"), make_study(design="case_control")])

    assert c is not None and c.level == "low"


def test_case_reports_or_news_only_are_very_low(make_study) -> None:
    studies = [make_study(design="case_report"), make_study(design="RCT", secondary_only=True)]

    c = certainty(studies)

    assert c is not None and c.level == "very_low"


def test_conflicting_directions_go_down_one(make_study) -> None:
    c = certainty([make_study(direction="-"), make_study(direction="null")])

    assert c is not None and c.level == "moderate"
    assert any("direction" in r for r in c.reasons)


def test_indirect_best_evidence_goes_down_one_more(make_study) -> None:
    studies = [
        make_study(direction="-", peco_match="indirect"),
        make_study(direction="+", peco_match="indirect"),
        make_study(design="cohort", direction="-"),  # direct, but not the best evidence
    ]

    c = certainty(studies)

    assert c is not None and c.level == "low"
    assert any("population or exposure" in r for r in c.reasons)


def test_floor_is_very_low(make_study) -> None:
    studies = [
        make_study(design="cohort", direction="+", peco_match="indirect"),
        make_study(design="cohort", direction="-", peco_match="indirect"),
    ]

    c = certainty(studies)

    assert c is not None and c.level == "very_low"


def test_excluded_and_retracted_studies_do_not_count(make_study) -> None:
    studies = [
        make_study(design="RCT", included=False, exclusion_reason="retracted", retracted=True),
        make_study(design="cohort"),
    ]

    c = certainty(studies)

    assert c is not None and c.level == "low"


def test_no_included_studies_has_no_certainty(make_study) -> None:
    assert certainty([make_study(included=False, exclusion_reason="off_topic")]) is None


def test_evidence_flow_counts_found_deduped_included_and_reasons(make_study) -> None:
    studies = [
        make_study(also_at=["https://news1", "https://news2"]),
        make_study(included=False, exclusion_reason="off_topic"),
        make_study(included=False, exclusion_reason="off_topic"),
        make_study(included=False, exclusion_reason="retracted", retracted=True),
    ]

    flow = evidence_flow(studies)

    assert (flow.found, flow.after_dedupe, flow.included) == (6, 4, 1)
    assert flow.excluded == {"off_topic": 2, "retracted": 1}


def test_direction_reason_is_readable(make_study) -> None:
    c = certainty([make_study(direction="-"), make_study(direction="null")])

    assert c is not None and "(-, null)" in c.reasons[-1]


def test_indirect_outliers_do_not_count_as_disagreement(make_study) -> None:
    # Live G1 (20261115-044257-37ecc1): 20 observational studies found no link; one indirect
    # "+" (mercury dose and several diagnoses) and one indirect "-" (sibling vaccination rates)
    # made it "very low", so the verdict became ❓. PRD §5.4 steps down for disagreement "with
    # no explanation"; being about another population or exposure is the explanation.
    studies = [
        *[make_study(design="cohort", direction="null") for _ in range(3)],
        make_study(design="case_control", direction="+", peco_match="indirect"),
        make_study(design="cohort", direction="-", peco_match="indirect"),
    ]

    c = certainty(studies)

    assert c is not None and c.level == "low"
    assert c.reasons == ["starts low: best evidence is observational"]


def test_direct_studies_that_disagree_still_go_down(make_study) -> None:
    studies = [
        make_study(design="cohort", direction="null"),
        make_study(design="cohort", direction="+"),
        make_study(design="cohort", direction="-", peco_match="indirect"),
    ]

    c = certainty(studies)

    assert c is not None and c.level == "very_low"
    assert "(+, null)" in c.reasons[-1]
