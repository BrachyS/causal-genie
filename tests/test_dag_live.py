"""Live check of the DAG prompt with the real large model. Run with `uv run pytest -m live`.

The fake-model tests in `test_dag.py` cover the harness; this one checks that the model itself
follows `prompts/dag.md` on the D1 licorice question.
"""

import re

import pytest
from openai import OpenAI

from causal_genie.agent.state import RunState
from causal_genie.agent.steps.dag import build_dag
from causal_genie.llm.client import LLMClient, load_models

pytestmark = pytest.mark.live

SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def test_d1_dag_follows_the_prompt(
    nebius: OpenAI, sample_state: RunState, confounder_claim, step_logs
) -> None:
    sample_state.claims.append(confounder_claim())
    llm = LLMClient(openai=nebius, models=load_models())

    edges = build_dag(sample_state, llm)

    by_pair = {(e.source, e.target): e for e in edges}
    question = by_pair[("licorice_gargle", "sore_throat")]
    assert "c1" in question.claim_ids, "the trial's effect claim should back the question edge"
    assert ("smoking", "sore_throat") in by_pair, "the adjusted-for variable should be a node"
    assert all(SNAKE.match(n) for e in edges for n in (e.source, e.target))
    assert len({n for e in edges for n in (e.source, e.target)}) <= 12
    # the harness logs every invented id, renamed question node and missing question edge
    assert not [r for r in step_logs.records if r.name.endswith("dag") and r.levelname == "WARNING"]
