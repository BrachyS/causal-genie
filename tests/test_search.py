from typing import cast

import pytest
from tavily import TavilyClient

from causal_genie.agent.state import CausalQuery
from causal_genie.tools.search import Searcher, thin_slice_queries


def test_search_returns_hits_and_records_call(fake_tavily) -> None:
    searcher = Searcher(cast(TavilyClient, fake_tavily))

    hits = searcher.search("licorice gargle sore throat")

    assert [h.url for h in hits] == ["https://pubmed.ncbi.nlm.nih.gov/23921656/"]
    assert fake_tavily.requests[0]["search_depth"] == "advanced"
    (call,) = searcher.calls
    assert call.query == "licorice gargle sore throat"
    assert call.hits == hits


def test_replay_returns_recorded_hits_without_tavily(fake_tavily) -> None:
    live = Searcher(cast(TavilyClient, fake_tavily))
    hits = live.search("licorice gargle sore throat")

    replay = Searcher.from_replay(live.calls)

    assert replay.search("licorice gargle sore throat") == hits
    with pytest.raises(KeyError, match="no recorded search"):
        replay.search("a new query")


def test_thin_slice_queries_name_exposure_outcome_and_design(
    sample_state,
) -> None:
    query = cast(CausalQuery, sample_state.query)

    queries = thin_slice_queries(query)

    assert len(queries) == 3
    assert all("licorice gargle" in q for q in queries)
    assert any("randomized" in q for q in queries)
    assert any("dataset" in q for q in queries)
