import hashlib
import json

import pytest
from conftest import GROUNDED, SNIPPET, epmc_result, extraction

from causal_genie.agent.state import ExtractCall, RunState, SearchCall, SearchHit
from causal_genie.agent.steps.sources import process_sources
from causal_genie.tools.search import Searcher

PUBMED = "https://pubmed.ncbi.nlm.nih.gov/23921656/"


def fixes(quotes: list[dict]) -> str:
    """Quote-fix model output: a new passage (or null) per field."""
    return json.dumps({"fixes": quotes})


def supported(*claim_ids: str, support: str = "supports") -> str:
    """Support-check model output judging every claim the same way."""
    return json.dumps({"judgments": [{"claim_id": c, "support": support} for c in claim_ids]})


@pytest.fixture
def state(sample_state: RunState) -> RunState:
    """The sample run, framed, with no studies yet."""
    return RunState(run_id="r1", question=sample_state.question, query=sample_state.query)


@pytest.fixture
def searcher() -> Searcher:
    """A searcher that has already found the D1 trial's PubMed page (snippet only)."""
    s = Searcher(tavily=None)
    hit = SearchHit(url=PUBMED, title="Licorice gargle trial", content=SNIPPET, score=0.9)
    s.calls.append(SearchCall(query="licorice gargle trial", hits=[hit], latency_s=0.1))
    return s


def sid(url: str) -> str:
    """The study id the harness gives a source's URL."""
    return "s" + hashlib.sha256(url.encode()).hexdigest()[:8]


def test_included_source_becomes_an_enriched_study_with_documented_claims(
    state, searcher, make_llm, make_epmc
) -> None:
    s = sid(PUBMED)
    llm, fake = make_llm([extraction(), supported(f"{s}-n", f"{s}-effect")])
    papers, served = make_epmc([epmc_result()])

    process_sources(state, llm, searcher, papers)

    (study,) = state.studies
    assert study.id == s
    assert study.doi == "10.1213/ane.0b013e318299a650"  # from Europe PMC, not the model
    assert "Randomized Controlled Trial" in study.publication_types
    assert served == ["EXT_ID:23921656 AND SRC:MED"]  # PMID read from the URL
    assert not study.full_text and not study.secondary_only
    assert {c.id: (c.kind, c.status) for c in state.claims} == {
        f"{s}-n": ("study_detail", "documented"),
        f"{s}-effect": ("effect_estimate", "documented"),
    }
    assert "Ignore" not in fake.requests[0]["messages"][0]["content"]  # prompt is the file
    assert "<<<SOURCE" in fake.requests[0]["messages"][1]["content"]


def test_quote_not_in_source_stays_unresolved_without_a_support_call(
    state, searcher, make_llm, make_epmc
) -> None:
    made_up = [{"field": "effect", "text": "licorice eliminated sore throat in every patient"}]
    llm, fake = make_llm([extraction(quotes=made_up), fixes(made_up)])  # fix repeats the error

    process_sources(state, llm, searcher, make_epmc([epmc_result()])[0])

    (claim,) = state.claims
    assert (claim.grounding, claim.support, claim.status) == (
        "quote_not_found",
        "unchecked",
        "unresolved",
    )
    assert len(fake.requests) == 2  # extraction + one quote fix; no support check


def test_missing_quote_is_sent_back_once_and_replaced(state, searcher, make_llm, make_epmc) -> None:
    copied = [{"field": "population", "text": "adults having surgery with a double-lumen tube"}]
    real = [{"field": "population", "text": "Randomized trial of 236 adults"}]
    claim = f"{sid(PUBMED)}-population"
    llm, fake = make_llm([extraction(quotes=copied), fixes(real), supported(claim)])

    process_sources(state, llm, searcher, make_epmc([epmc_result()])[0])

    assert state.studies[0].quotes == {"population": "Randomized trial of 236 adults"}
    (c,) = state.claims
    assert (c.grounding, c.status) == ("quote_found", "documented")
    fix_input = fake.requests[1]["messages"][1]["content"]
    assert "adults having surgery with a double-lumen tube" in fix_input and SNIPPET in fix_input


def test_quote_the_fix_cannot_find_is_dropped(state, searcher, make_llm, make_epmc) -> None:
    copied = [{"field": "population", "text": "adults having surgery with a double-lumen tube"}]
    llm, fake = make_llm(
        [extraction(quotes=copied), fixes([{"field": "population", "text": None}])]
    )

    process_sources(state, llm, searcher, make_epmc([epmc_result()])[0])

    assert state.studies[0].quotes == {} and state.claims == []
    assert len(fake.requests) == 2


def test_source_comes_before_the_question(state, searcher, make_llm, make_epmc) -> None:
    llm, fake = make_llm([extraction(quotes=[])])

    process_sources(state, llm, searcher, make_epmc([epmc_result()])[0])

    user = fake.requests[0]["messages"][1]["content"]
    assert user.index("<<<SOURCE") < user.index("Population:")


def test_unsupported_quote_is_unresolved(state, searcher, make_llm, make_epmc) -> None:
    claim = f"{sid(PUBMED)}-n"
    llm, _ = make_llm([extraction(quotes=GROUNDED[:1]), supported(claim, support="contradicts")])

    process_sources(state, llm, searcher, make_epmc([epmc_result()])[0])

    (claim,) = state.claims
    assert (claim.grounding, claim.support, claim.status) == (
        "quote_found",
        "contradicts",
        "unresolved",
    )


def test_excluded_source_is_neither_enriched_nor_checked(
    state, searcher, make_llm, make_epmc
) -> None:
    llm, fake = make_llm([extraction(included=False)])
    papers, served = make_epmc([epmc_result()])

    process_sources(state, llm, searcher, papers)

    (study,) = state.studies
    assert study.exclusion_reason == "off_topic"
    assert served == [] and state.claims == [] and len(fake.requests) == 1


def test_retracted_paper_is_excluded(state, searcher, make_llm, make_epmc) -> None:
    llm, fake = make_llm([extraction()])
    types = {"pubType": ["Retracted Publication"]}

    process_sources(state, llm, searcher, make_epmc([epmc_result(pubTypeList=types)])[0])

    (study,) = state.studies
    assert study.retracted and not study.included and study.exclusion_reason == "retracted"
    assert state.claims == [] and len(fake.requests) == 1


def test_sources_already_read_are_not_read_again(state, searcher, make_llm, make_epmc) -> None:
    llm, fake = make_llm([extraction(quotes=[])])
    papers = make_epmc([epmc_result()])[0]

    process_sources(state, llm, searcher, papers)
    process_sources(state, llm, searcher, papers)

    assert len(fake.requests) == 1 and len(state.studies) == 1


def test_full_text_replaces_the_snippet_reading(state, searcher, make_llm, make_epmc) -> None:
    llm, fake = make_llm([extraction(quotes=[]), extraction(quotes=[], n=235)])
    papers = make_epmc([epmc_result()])[0]
    process_sources(state, llm, searcher, papers)

    searcher.extracts.append(ExtractCall(url=PUBMED, content=SNIPPET, error=None, latency_s=0.1))
    process_sources(state, llm, searcher, papers)

    (study,) = state.studies
    assert study.full_text and study.n == 235
    assert "full page text" in fake.requests[1]["messages"][1]["content"]


def test_news_story_merges_into_the_paper_found_later(state, make_llm, make_epmc) -> None:
    news_url = "https://news.example.com/licorice"
    searcher = Searcher(tavily=None)
    hit = SearchHit(url=news_url, title="Licorice for sore throat", content=SNIPPET, score=0.5)
    searcher.calls.append(SearchCall(query="q1", hits=[hit], latency_s=0.1))
    news = extraction(source_kind="secondary", quotes=GROUNDED[:1])
    llm, _ = make_llm([news, supported(f"{sid(news_url)}-n"), extraction(quotes=[])])
    papers = make_epmc([epmc_result()])[0]  # the title lookup finds the paper → same DOI
    process_sources(state, llm, searcher, papers)

    hit = SearchHit(url=PUBMED, title="Licorice gargle trial", content=SNIPPET, score=0.9)
    searcher.calls.append(SearchCall(query="q2", hits=[hit], latency_s=0.1))
    process_sources(state, llm, searcher, papers)

    (study,) = state.studies
    assert study.url == PUBMED and study.also_at == [news_url] and not study.secondary_only
    assert state.claims == []  # the news story's claims went with it


def test_quotes_for_empty_fields_are_ignored(state, searcher, make_llm, make_epmc) -> None:
    quotes = [{"field": "adjusted_for", "text": "No confounders are mentioned."}]
    llm, fake = make_llm([extraction(quotes=quotes)])

    process_sources(state, llm, searcher, make_epmc([epmc_result()])[0])

    assert state.studies[0].quotes == {} and state.claims == []
    assert len(fake.requests) == 1
