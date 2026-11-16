import pytest

from causal_genie.agent.certainty import certainty
from causal_genie.agent.state import (
    DagEdge,
    DatasetCandidate,
    DesignCheck,
    Mapping,
    NodeMapping,
    Result,
    Route,
    RunState,
)
from causal_genie.agent.verdict import pick_verdict, unmeasured_confounders


def node(name: str, *columns: str) -> NodeMapping:
    return NodeMapping(
        node=name, columns=list(columns), derive=None, cutoff=None, categorical=False
    )


def candidate(access: str = "open", accepted: bool = False) -> DatasetCandidate:
    return DatasetCandidate.model_validate(
        {
            "url": "https://example.org/data.csv",
            "host": "example.org",
            "license": None,
            "access": access,
            "coverage": {},
            "score": None,
            "rejected_reason": None if accepted else "lacks sore_throat",
            "source": "search",
            "title": "t",
            "description": "d",
            "accepted": accepted,
        }
    )


def estimate(tool: str) -> Result:
    return Result(id="res-1", tool=tool, tool_version="0", inputs_hash="h", outputs={})


@pytest.fixture
def researched(sample_state: RunState) -> RunState:
    """The sample run after research: one included RCT, certainty set, no data yet."""
    sample_state.certainty = certainty(sample_state.studies)
    return sample_state


@pytest.fixture
def observational(researched: RunState, confounder_claim) -> RunState:
    """An observational estimate from a table holding exposure, outcome and smoking."""
    researched.claims.append(confounder_claim())
    researched.studies[0].adjusted_for = ["smoking"]
    researched.edges += [
        DagEdge(source="smoking", target="licorice_gargle", claim_ids=["s1-adjusted_for"]),
        DagEdge(source="smoking", target="sore_throat", claim_ids=["s1-adjusted_for"]),
    ]
    researched.mapping = Mapping(
        nodes=[
            node("licorice_gargle", "treat"),
            node("sore_throat", "pain"),
            node("smoking", "preOp_smoking"),
        ],
        unit=None,
        time=None,
    )
    researched.datasets = [candidate(accepted=True)]
    researched.results = [estimate("backdoor_effect")]
    return researched


def test_not_causal(researched: RunState) -> None:
    researched.route = Route(kind="not_causal", run_id=None, reason="r", causal_rewrite=None)

    assert pick_verdict(researched) == "not_causal"


def test_randomized_estimate_is_estimated(researched: RunState) -> None:
    researched.results = [estimate("rct_effect")]

    assert pick_verdict(researched) == "estimated"


def test_observational_estimate_with_every_named_confounder_is_estimated(observational) -> None:
    assert unmeasured_confounders(observational) == []
    assert pick_verdict(observational) == "estimated"


def test_confounder_named_by_a_study_but_not_in_the_data_is_strong_assumptions(
    observational: RunState,
) -> None:
    observational.studies[0].adjusted_for = ["smoking status", "body mass index"]

    assert unmeasured_confounders(observational) == ["body mass index"]
    assert pick_verdict(observational) == "estimated_strong_assumptions"


def test_unchecked_confounder_claims_do_not_count(observational: RunState) -> None:
    observational.studies[0].adjusted_for = ["body mass index"]
    observational.claims[-1].status = "unresolved"

    assert pick_verdict(observational) == "estimated"


def test_dag_confounder_left_unmapped_is_strong_assumptions(observational: RunState) -> None:
    assert observational.mapping is not None
    observational.mapping.nodes[2] = node("smoking")  # the table lacks it

    assert unmeasured_confounders(observational) == ["smoking"]
    assert pick_verdict(observational) == "estimated_strong_assumptions"


@pytest.mark.parametrize("rule", ["full_moon", "day_of_week", "weekend", "holiday", "season"])
def test_an_exposure_read_off_the_calendar_has_no_confounders(observational, rule: str) -> None:
    # Live D0 (run 20261116-034032-d7d345): studies of full-moon births adjusted for parity and
    # gestational age, which cannot change the moon's phase; they made the verdict ⚠️.
    assert observational.mapping is not None
    observational.mapping.nodes[0] = node("licorice_gargle", "date").model_copy(
        update={"derive": rule}
    )
    observational.studies[0].adjusted_for = ["parity", "gestational age"]

    assert unmeasured_confounders(observational) == []
    assert pick_verdict(observational) == "estimated"


def test_an_exposure_from_a_cutoff_can_still_be_confounded(observational: RunState) -> None:
    assert observational.mapping is not None
    observational.mapping.nodes[0] = node("licorice_gargle", "dose").model_copy(
        update={"derive": "cutoff", "cutoff": 0}
    )
    observational.studies[0].adjusted_for = ["body mass index"]

    assert unmeasured_confounders(observational) == ["body mass index"]


def test_randomized_estimate_ignores_unmeasured_confounders(observational) -> None:
    observational.studies[0].adjusted_for = ["body mass index"]
    observational.results = [estimate("rct_effect")]

    assert pick_verdict(observational) == "estimated"


def test_design_that_rules_out_an_estimate_is_study_plan(researched: RunState) -> None:
    researched.datasets = [candidate(accepted=True)]
    researched.design = DesignCheck(
        design="selection_after_assignment",
        estimable=False,
        key_assumption="k",
        estimand="none",
        check="c",
        reasons=["r"],
    )

    assert pick_verdict(researched) == "study_plan"


def test_only_restricted_data_is_restricted(researched: RunState) -> None:
    researched.datasets = [candidate(access="restricted"), candidate()]

    assert pick_verdict(researched) == "restricted"


def test_restricted_candidate_does_not_matter_once_a_table_was_accepted(researched) -> None:
    researched.datasets = [candidate(access="restricted"), candidate(accepted=True)]

    assert pick_verdict(researched) == "published_only"  # e.g. the exposure was not mapped


def test_studies_but_no_data_is_published_only(researched: RunState) -> None:
    researched.datasets = [candidate()]

    assert pick_verdict(researched) == "published_only"


def test_no_included_study_is_insufficient(researched: RunState) -> None:
    researched.studies[0].included = False
    researched.studies[0].exclusion_reason = "off_topic"
    researched.certainty = certainty(researched.studies)

    assert pick_verdict(researched) == "insufficient"


def test_very_low_certainty_is_insufficient(researched: RunState) -> None:
    researched.studies[0].design = "case_report"
    researched.certainty = certainty(researched.studies)

    assert researched.certainty is not None and researched.certainty.level == "very_low"
    assert pick_verdict(researched) == "insufficient"


def test_unmeasured_confounders_only_matter_for_an_observational_estimate(observational):
    observational.studies[0].adjusted_for = ["body mass index"]
    observational.results = [estimate("rct_effect")]

    assert unmeasured_confounders(observational) == []
