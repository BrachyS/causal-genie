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
from causal_genie.agent.verdict import (
    cited_paths,
    common_causes,
    pick_verdict,
    unmeasured_confounders,
)


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


@pytest.fixture
def spurious(researched: RunState) -> RunState:
    """Ice cream and shark bites: no study tests the link itself; a blog post (checked
    quote) says hot weather drives both."""
    blog = researched.studies[0]
    blog.design, blog.strategy, blog.peco_match = "other", "other", "indirect"
    blog.secondary_only = True
    researched.certainty = certainty(researched.studies)
    researched.edges[0].question = True  # as in real runs: the hypothesis, not cited evidence
    researched.edges += [
        DagEdge(source="temperature", target="licorice_gargle", claim_ids=["c1"]),
        DagEdge(source="temperature", target="sore_throat", claim_ids=["c1"]),
    ]
    return researched


def test_a_link_no_study_tests_but_a_cited_common_cause_explains_is_common_cause(
    spurious: RunState,
) -> None:
    assert spurious.certainty is not None and spurious.certainty.level == "very_low"
    assert common_causes(spurious) == ["temperature"]
    assert pick_verdict(spurious) == "common_cause"


def test_an_assumed_common_cause_does_not_count(spurious: RunState) -> None:
    for e in spurious.edges:
        if e.source == "temperature":
            e.claim_ids = []

    assert common_causes(spurious) == []
    assert pick_verdict(spurious) == "insufficient"


def test_a_common_cause_cited_only_by_unchecked_claims_does_not_count(spurious) -> None:
    spurious.claims[0].status = "unresolved"

    assert pick_verdict(spurious) == "insufficient"


def test_a_common_cause_needs_edges_into_both_exposure_and_outcome(spurious) -> None:
    spurious.edges = [e for e in spurious.edges if e.target != "sore_throat"]

    assert pick_verdict(spurious) == "insufficient"


def test_a_common_cause_can_act_through_a_chain_of_cited_edges(spurious: RunState) -> None:
    """Live run: hot weather -> ice cream sales, and hot weather -> beach attendance ->
    shark bites; the common cause is one step back from the outcome."""
    spurious.edges = [e for e in spurious.edges if e.source != "temperature"] + [
        DagEdge(source="temperature", target="licorice_gargle", claim_ids=["c1"]),
        DagEdge(source="temperature", target="crowds", claim_ids=["c1"]),
        DagEdge(source="crowds", target="sore_throat", claim_ids=["c1"]),
    ]

    assert common_causes(spurious) == ["temperature"]
    assert pick_verdict(spurious) == "common_cause"


def test_a_chain_with_an_assumed_link_does_not_count(spurious: RunState) -> None:
    spurious.edges = [e for e in spurious.edges if e.source != "temperature"] + [
        DagEdge(source="temperature", target="licorice_gargle", claim_ids=["c1"]),
        DagEdge(source="temperature", target="crowds", claim_ids=[]),
        DagEdge(source="crowds", target="sore_throat", claim_ids=["c1"]),
    ]

    assert common_causes(spurious) == []


def test_a_cited_path_from_the_exposure_to_the_outcome_rules_out_common_cause(spurious) -> None:
    """Question 6 (Nov 22): family income drove both school rating and admission, but cited
    edges also ran rating -> academic preparation -> admission. Evidence that the exposure
    acts through a mediator is not "explained by a common cause"."""
    spurious.edges += [
        DagEdge(source="licorice_gargle", target="mucosa", claim_ids=["c1"]),
        DagEdge(source="mucosa", target="sore_throat", claim_ids=["c1"]),
    ]

    assert common_causes(spurious) == []
    assert pick_verdict(spurious) == "insufficient"


def test_cited_paths_lists_each_path_from_the_exposure_to_the_outcome(spurious) -> None:
    spurious.edges += [
        DagEdge(source="licorice_gargle", target="mucosa", claim_ids=["c1"]),
        DagEdge(source="mucosa", target="sore_throat", claim_ids=["c1"]),
        DagEdge(source="licorice_gargle", target="cough", claim_ids=["c1"]),
        DagEdge(source="cough", target="sore_throat", claim_ids=[]),  # assumed: not a path
    ]

    assert cited_paths(spurious) == [["licorice_gargle", "mucosa", "sore_throat"]]


def test_the_question_edge_is_not_a_cited_path(spurious) -> None:
    assert cited_paths(spurious) == []


def test_a_path_from_the_exposure_with_an_assumed_link_keeps_common_cause(spurious) -> None:
    spurious.edges += [
        DagEdge(source="licorice_gargle", target="mucosa", claim_ids=["c1"]),
        DagEdge(source="mucosa", target="sore_throat", claim_ids=[]),
    ]

    assert common_causes(spurious) == ["temperature"]


@pytest.mark.parametrize("design", ["RCT", "cohort"])
def test_a_direct_study_of_the_link_rules_out_common_cause(spurious, design: str) -> None:
    """Confounders exist for real effects too: once a trial or an observational study of
    this exposure and outcome is in, the evidence speaks for itself."""
    study = spurious.studies[0]
    study.design, study.peco_match, study.secondary_only = design, "direct", False
    spurious.certainty = certainty(spurious.studies)

    assert common_causes(spurious) == []
    assert pick_verdict(spurious) == "published_only"


def test_an_estimate_beats_common_cause(spurious: RunState) -> None:
    spurious.results = [estimate("rct_effect")]

    assert pick_verdict(spurious) == "estimated"


def test_unmeasured_confounders_only_matter_for_an_observational_estimate(observational):
    observational.studies[0].adjusted_for = ["body mass index"]
    observational.results = [estimate("rct_effect")]

    assert unmeasured_confounders(observational) == []
