from typing import cast

import pytest
from tavily import TavilyClient

from causal_genie.tools.search import Searcher


@pytest.fixture
def searcher(fake_tavily) -> Searcher:
    """A live searcher over the fake Tavily."""
    return Searcher(cast(TavilyClient, fake_tavily))


def test_search_returns_hits_and_records_call(searcher, fake_tavily) -> None:
    hits = searcher.search("licorice gargle sore throat")

    assert [h.url for h in hits] == ["https://pubmed.ncbi.nlm.nih.gov/23921656/"]
    assert fake_tavily.requests[0]["search_depth"] == "advanced"
    (call,) = searcher.calls
    assert call.query == "licorice gargle sore throat"
    assert call.hits == hits


def test_replay_returns_recorded_hits_without_tavily(searcher) -> None:
    hits = searcher.search("licorice gargle sore throat")

    replay = Searcher.from_replay(searcher.calls)

    assert replay.search("licorice gargle sore throat") == hits
    with pytest.raises(KeyError, match="no recorded search"):
        replay.search("a new query")


def test_extract_returns_full_text_and_records_call(searcher) -> None:
    call = searcher.extract("https://example.org/paper")

    assert call.content == "Full text of the paper."
    assert call.error is None
    assert searcher.extracts == [call]


def test_failed_extract_is_recorded_not_raised(searcher) -> None:
    call = searcher.extract("https://example.org/fail")

    assert call.content is None
    assert call.error == "blocked"


def test_extract_replays_without_tavily(searcher) -> None:
    live = searcher.extract("https://example.org/paper")

    replay = Searcher.from_replay([], extracts=searcher.extracts)

    assert replay.extract("https://example.org/paper") == live
    with pytest.raises(KeyError, match="no recorded extract"):
        replay.extract("https://example.org/other")
