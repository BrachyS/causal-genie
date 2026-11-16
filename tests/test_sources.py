import hashlib
import json
import logging

import pytest
from conftest import GROUNDED, SNIPPET, epmc_result, extraction

from causal_genie.agent.certainty import evidence_flow
from causal_genie.agent.state import ExtractCall, Extraction, RunState, SearchCall, SearchHit
from causal_genie.agent.steps.sources import is_variable_name, process_sources
from causal_genie.llm.client import PROMPTS_DIR, LLMClient
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


NOISE = [  # adjusted_for entries extracted in live D0 run 20261114-154259-6bb1d2
    "non linearly the effect of several confounding variables",
    "quasi-periodic and discrete components",
]


@pytest.mark.parametrize(
    ("item", "kept"),
    [
        (NOISE[0], False),  # "effect", "confounding", "variables"
        (NOISE[1], False),  # "components"
        ("other confounders", False),
        ("covariates", False),
        ("maternal age", True),
        ("parity", True),
        ("body mass index", True),
        ("use of oral contraceptives", True),
        ("comorbidities such as sleep apnea and insomnia", True),  # long but real
        ("", False),
        ("smoking_status", True),
    ],
)
def test_only_variable_names_count_as_adjusted_for(item: str, kept: bool) -> None:
    assert is_variable_name(item) is kept


def test_adjusted_for_descriptions_are_dropped_with_their_claim(
    state, searcher, make_llm, make_epmc, caplog
) -> None:
    quotes = [{"field": "adjusted_for", "text": "Randomized trial of 236 adults"}]
    llm, fake = make_llm([extraction(quotes=quotes, adjusted_for=NOISE)])

    with caplog.at_level(logging.INFO):
        process_sources(state, llm, searcher, make_epmc([epmc_result()])[0])

    (study,) = state.studies
    assert study.adjusted_for == [] and study.quotes == {} and state.claims == []
    assert len(fake.requests) == 1  # no support check: nothing left to check
    assert all(item in caplog.text for item in NOISE)


def test_real_adjusted_for_names_survive_next_to_dropped_ones(
    state, searcher, make_llm, make_epmc
) -> None:
    quotes = [{"field": "adjusted_for", "text": "Randomized trial of 236 adults"}]
    outputs = [
        extraction(quotes=quotes, adjusted_for=["maternal age", NOISE[1]]),
        supported(f"{sid(PUBMED)}-adjusted_for"),
    ]
    llm, _ = make_llm(outputs)

    process_sources(state, llm, searcher, make_epmc([epmc_result()])[0])

    (study,) = state.studies
    assert study.adjusted_for == ["maternal age"]
    (claim,) = state.claims
    assert claim.text.endswith("adjusted_for = maternal age")


def test_failed_extraction_skips_the_source_not_the_run(
    state, searcher, make_llm, make_epmc, prompts_dir, caplog
) -> None:
    # Live G1 (20261115-034156-f3618c): an extract of a Wikipedia page over-quoted past its
    # 1,500-token cap twice, and the schema failure ended the run (user chose to skip).
    llm, fake = make_llm(["not json", "still not json"])
    papers, served = make_epmc([epmc_result()])

    with caplog.at_level(logging.ERROR):
        process_sources(state, llm, searcher, papers)

    (study,) = state.studies
    assert (study.id, study.url, study.title) == (sid(PUBMED), PUBMED, "Licorice gargle trial")
    assert not study.included and study.exclusion_reason == "extraction_failed"
    assert served == [] and state.claims == [] and len(fake.requests) == 2  # call + repair
    assert evidence_flow(state.studies).excluded == {"extraction_failed": 1}
    assert "extraction failed" in caplog.text

    replayed = RunState(run_id="r2", question=state.question, query=state.query)
    process_sources(
        replayed, LLMClient.from_replay(llm.calls, prompts_dir), searcher, make_epmc([])[0]
    )
    assert replayed.studies == state.studies


def test_extraction_failed_is_never_offered_to_the_model() -> None:
    schema = json.dumps(Extraction.model_json_schema())
    assert "extraction_failed" not in schema


def test_extract_prompt_reads_direction_from_the_interval_not_the_wording() -> None:
    """Certainty steps down when best studies point different ways, so a non-significant
    result the authors call "a slight reduction" must not count as `-` (live G1, run
    20261115-053439-dc8356: HR 0.93, 95% CI 0.85-1.02 extracted as `-`)."""
    text = (PROMPTS_DIR / "extract.md").read_text()

    assert "interval includes no effect" in text
    assert "1 for a ratio" in text and "0 for a difference" in text
