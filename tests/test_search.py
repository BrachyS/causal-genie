from typing import cast

import pytest
from tavily import TavilyClient

from causal_genie.agent.state import SearchCall
from causal_genie.tools.search import RESEARCH_DOMAINS, Searcher


@pytest.fixture
def searcher(fake_tavily) -> Searcher:
    """A live searcher over the fake Tavily."""
    return Searcher(cast(TavilyClient, fake_tavily))


def test_search_returns_hits_and_records_call(searcher, fake_tavily) -> None:
    hits = searcher.search("licorice gargle sore throat")

    assert [h.url for h in hits] == ["https://pubmed.ncbi.nlm.nih.gov/23921656/"]
    assert fake_tavily.requests[0]["search_depth"] == "advanced"
    assert fake_tavily.requests[0]["max_results"] == 10  # same credit cost as 5; more recall
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


def test_search_without_domains_sends_no_domain_filter(searcher, fake_tavily) -> None:
    searcher.search("licorice gargle sore throat")

    assert "include_domains" not in fake_tavily.requests[0]
    assert searcher.calls[0].domains == []


def test_preferred_domains_are_sent_in_prefer_mode_and_recorded(searcher, fake_tavily) -> None:
    searcher.search("licorice gargle sore throat", domains=RESEARCH_DOMAINS)

    request = fake_tavily.requests[0]
    assert request["include_domains"] == list(RESEARCH_DOMAINS)
    assert request["include_domains_mode"] == "prefer"  # never restrict: news stays a lead
    assert searcher.calls[0].domains == list(RESEARCH_DOMAINS)


def test_replay_matches_on_query_and_domains(searcher) -> None:
    searcher.search("licorice gargle sore throat", domains=RESEARCH_DOMAINS)

    replay = Searcher.from_replay(searcher.calls)

    assert replay.search("licorice gargle sore throat", domains=RESEARCH_DOMAINS)
    with pytest.raises(KeyError, match="no recorded search"):
        replay.search("licorice gargle sore throat")  # same text, no domains: another search


def test_runs_saved_before_domains_still_load() -> None:
    call = SearchCall.model_validate({"query": "q", "hits": [], "latency_s": 0.1})

    assert call.domains == []
