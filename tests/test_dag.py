import json

from causal_genie.agent.state import RunState
from causal_genie.agent.steps.dag import build_dag


def proposal(*edges: tuple[str, str, list[str]]) -> str:
    """DAG model output with the given (source, target, claim_ids) edges."""
    return json.dumps(
        {"edges": [{"source": s, "target": t, "claim_ids": ids} for s, t, ids in edges]}
    )


def test_edges_cite_ledger_claims_from_the_large_model(
    make_llm, sample_state: RunState, confounder_claim
) -> None:
    sample_state.claims.append(confounder_claim())
    llm, fake = make_llm(
        [
            proposal(
                ("licorice_gargle", "sore_throat", ["c1"]),
                ("smoking", "sore_throat", ["s1-adjusted_for"]),
            )
        ]
    )

    edges = build_dag(sample_state, llm)

    assert [(e.source, e.target, e.claim_ids) for e in edges] == [
        ("licorice_gargle", "sore_throat", ["c1"]),
        ("smoking", "sore_throat", ["s1-adjusted_for"]),
    ]
    assert fake.requests[0]["model"] == "test/large"
    sent = fake.requests[0]["messages"][1]["content"]
    assert "licorice_gargle" in sent and "sore_throat" in sent
    assert "s1-adjusted_for" in sent and "smokers were balanced" in sent


def test_edge_citing_an_unknown_claim_is_marked_assumed(
    make_llm, sample_state: RunState, step_logs
) -> None:
    llm, _ = make_llm(
        [
            proposal(
                ("licorice_gargle", "sore_throat", ["c1", "c99"]),
                ("age", "sore_throat", ["c42"]),
            )
        ]
    )

    edges = build_dag(sample_state, llm)

    assert edges[0].claim_ids == ["c1"]
    assert edges[1].claim_ids == [] and edges[1].assumed
    assert "c99" in step_logs.text and "c42" in step_logs.text


def test_unresolved_claims_are_not_shown_and_cannot_be_cited(
    make_llm, sample_state: RunState, confounder_claim
) -> None:
    sample_state.claims.append(confounder_claim(status="unresolved"))
    llm, fake = make_llm(
        [
            proposal(
                ("licorice_gargle", "sore_throat", ["c1"]),
                ("smoking", "sore_throat", ["s1-adjusted_for"]),
            )
        ]
    )

    edges = build_dag(sample_state, llm)

    assert "s1-adjusted_for" not in fake.requests[0]["messages"][1]["content"]
    assert edges[1].assumed


def test_missing_exposure_outcome_edge_is_added_as_assumed(
    make_llm, sample_state: RunState
) -> None:
    llm, _ = make_llm([proposal(("age", "sore_throat", []))])

    edges = build_dag(sample_state, llm)

    question_edge = [e for e in edges if (e.source, e.target) == ("licorice_gargle", "sore_throat")]
    assert len(question_edge) == 1 and question_edge[0].assumed


def test_near_copy_of_exposure_name_is_logged(make_llm, sample_state: RunState, step_logs) -> None:
    llm, _ = make_llm(
        [
            proposal(
                ("licorice_gargling", "sore_throat", ["c1"]),
                ("sore_throat_severity", "sore_throat", []),
            )
        ]
    )

    build_dag(sample_state, llm)

    renamed = [r.message for r in step_logs.records if "looks like" in r.message]
    assert len(renamed) == 1 and "licorice_gargling" in renamed[0]
