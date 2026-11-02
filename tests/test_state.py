from pathlib import Path

import pytest
from pydantic import ValidationError

from causal_genie.agent.state import ResearchPlan, RunState


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
