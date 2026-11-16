import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from causal_genie.agent.state import (
    Adjustment,
    DagEdge,
    DagProposal,
    DatasetCandidate,
    LookupCall,
    ResearchPlan,
    RunState,
)


def test_run_state_round_trips_through_json(tmp_path: Path, sample_state: RunState) -> None:
    path = sample_state.save(tmp_path)

    assert path == tmp_path / "r1" / "state.json"
    assert RunState.load(path) == sample_state


def test_saved_state_ends_with_one_newline(tmp_path: Path, sample_state: RunState) -> None:
    # gallery copies are committed; the end-of-file pre-commit hook rewrites a file without one
    text = sample_state.save(tmp_path).read_text()

    assert text.endswith("}\n") and not text.endswith("\n\n")


def test_run_state_totals_tokens_and_latency(sample_state: RunState) -> None:
    sample_state.llm_calls.append(sample_state.llm_calls[0].model_copy())

    assert sample_state.tokens == {"in": 20, "out": 10}
    assert sample_state.llm_latency_s == pytest.approx(0.2)


def test_model_output_schemas_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(
            {"queries": [], "done": True, "done_reason": "enough", "verdict": "✅"}
        )


def test_question_edge_is_not_assumed_without_claims() -> None:
    question = DagEdge(source="full_moon", target="births", claim_ids=[], question=True)

    assert not question.assumed
    assert DagEdge(source="weekend", target="births", claim_ids=[]).assumed


def test_old_saved_edges_load_as_non_question_edges() -> None:
    edge = DagEdge.model_validate({"source": "a", "target": "b", "claim_ids": []})

    assert edge.question is False


def test_model_is_not_shown_the_question_flag() -> None:
    assert "question" not in json.dumps(DagProposal.model_json_schema())


def test_runs_saved_before_t10_load_with_empty_new_fields() -> None:
    adjustment = Adjustment.model_validate({"adjust_for": [], "missing": [], "randomized": False})
    candidate = DatasetCandidate.model_validate(
        {
            "url": "https://x/y.csv",
            "host": "x",
            "license": None,
            "access": "open",
            "coverage": {},
            "score": None,
            "rejected_reason": None,
            "source": "search",
            "title": "t",
            "description": "d",
        }
    )

    assert adjustment.precision == []
    assert candidate.study_id is None


def test_dag_proposal_lists_exogenous_nodes_and_defaults_to_none() -> None:
    proposed = DagProposal.model_validate(
        {"edges": [], "exogenous": ["full_moon"]}  # nothing in the graph causes moon phase
    )
    assert proposed.exogenous == ["full_moon"]
    assert DagProposal.model_validate({"edges": []}).exogenous == []  # older recorded runs
    assert "exogenous" in DagProposal.model_json_schema()["properties"]


def test_runs_saved_before_t7_6_load_with_no_exogenous_nodes(sample_state: RunState) -> None:
    saved = sample_state.model_dump(exclude={"exogenous", "exogenous_dropped"})
    loaded = RunState.model_validate(saved)
    assert loaded.exogenous == [] and loaded.exogenous_dropped == []


def test_lookup_call_records_a_failure_and_old_lookups_load_without_one() -> None:
    old = LookupCall.model_validate(
        {"query": "EXT_ID:1 AND SRC:MED", "record": None, "latency_s": 0.1}
    )
    assert old.error is None

    failed = LookupCall(query="q", record=None, latency_s=1.0, error="HTTP 500")
    assert LookupCall.model_validate_json(failed.model_dump_json()) == failed
