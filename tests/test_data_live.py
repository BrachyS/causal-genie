"""Live check of data discovery on D0 and D1 (T8 "done when"). Run with `uv run pytest -m live`.

Real Tavily (restricted to allowlisted hosts), the real small model, Rdatasets and downloads.
The studies' data links are cleared, so the data must be found by search alone.
"""

from pathlib import Path

import httpx
import pytest
from openai import OpenAI
from tavily import TavilyClient

from causal_genie.agent.state import DagEdge, RunState, Variable
from causal_genie.agent.steps.data import find_data
from causal_genie.llm.client import LLMClient, load_models
from causal_genie.tools.download import Fetcher
from causal_genie.tools.search import Searcher

pytestmark = pytest.mark.live


@pytest.fixture
def d0_state(sample_state: RunState) -> RunState:
    """The full-moon question with a small DAG of calendar confounders."""
    assert sample_state.query is not None
    query = sample_state.query.model_copy(
        update={
            "population": "births in the United States",
            "exposure": Variable(name="full_moon", definition="day of a full moon"),
            "comparator": "other days of the lunar cycle",
            "outcome": Variable(name="births", definition="number of births per day"),
            "deviation_strategy": "n/a",
        }
    )
    edges = [
        DagEdge(source="full_moon", target="births", claim_ids=[]),
        DagEdge(source="weekend", target="births", claim_ids=[]),
        DagEdge(source="holiday", target="births", claim_ids=[]),
    ]
    return RunState(
        run_id="live-d0", question="Does the full moon cause more births?", query=query, edges=edges
    )


def find(state: RunState, nebius: OpenAI, tavily: TavilyClient, tmp_path: Path):
    """Run data discovery live and return the accepted candidate."""
    llm = LLMClient(openai=nebius, models=load_models())
    with httpx.Client(timeout=60) as http:
        found = find_data(state, llm, Searcher(tavily), Fetcher(http), tmp_path)
    return found, llm


def test_d1_trial_data_found_by_search(
    nebius: OpenAI, tavily: TavilyClient, sample_state: RunState, tmp_path: Path
) -> None:
    for s in sample_state.studies:
        s.data_link = None

    found, llm = find(sample_state, nebius, tavily, tmp_path)

    assert found is not None, [(c.url, c.rejected_reason) for c in sample_state.datasets]
    assert "licorice_gargle" in found.url
    assert found.coverage["licorice_gargle"] == "treat"
    assert sample_state.profile is not None and sample_state.profile.rows == 235
    print(found.rank, found.url, llm.calls[-1].tokens_in, llm.calls[-1].tokens_out)


def test_d0_daily_births_found_by_search(
    nebius: OpenAI, tavily: TavilyClient, d0_state: RunState, tmp_path: Path
) -> None:
    found, llm = find(d0_state, nebius, tavily, tmp_path)

    assert found is not None, [(c.url, c.rejected_reason) for c in d0_state.datasets]
    assert "birth" in found.url.lower()
    assert d0_state.profile is not None and d0_state.profile.rows >= 365  # one row per day
    print(found.rank, found.url, found.coverage, llm.calls[-1].tokens_in)
