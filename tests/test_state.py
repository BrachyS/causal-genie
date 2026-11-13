import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from causal_genie.agent.state import (
    Adjustment,
    DagEdge,
    DagProposal,
    DatasetCandidate,
    ResearchPlan,
    RunState,
)


def test_run_state_round_trips_through_json(tmp_path: Path, sample_state: RunState) -> None:
    path = sample_state.save(tmp_path)

    assert path == tmp_path / "r1" / "state.json"
    assert RunState.load(path) == sample_state


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
