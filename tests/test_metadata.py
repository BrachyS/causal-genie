import logging

import httpx
import pytest
from conftest import D1_TITLE, epmc_result

from causal_genie.tools.metadata import EuropePMC, ids_from_url, titles_match


@pytest.mark.parametrize(
    ("url", "ids"),
    [
        ("https://pubmed.ncbi.nlm.nih.gov/23921656/", {"pmid": "23921656"}),
        ("https://www.ncbi.nlm.nih.gov/pubmed/23921656", {"pmid": "23921656"}),
        ("https://europepmc.org/article/MED/23921656", {"pmid": "23921656"}),
        ("https://pmc.ncbi.nlm.nih.gov/articles/PMC4123456/", {"pmcid": "PMC4123456"}),
        ("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4123456", {"pmcid": "PMC4123456"}),
        ("https://doi.org/10.1213/ANE.0b013e318299a650", {"doi": "10.1213/ANE.0b013e318299a650"}),
        (
            "https://onlinelibrary.wiley.com/doi/full/10.1111/anae.12345/abstract",
            {"doi": "10.1111/anae.12345"},
        ),
        ("https://www.bbc.com/news/health-123", {}),
    ],
)
def test_ids_from_url(url: str, ids: dict) -> None:
    assert ids_from_url(url) == ids


def test_lookup_by_pmid_reads_doi_author_year_and_types(make_epmc) -> None:
    epmc, served = make_epmc([epmc_result()])

    record = epmc.find(pmid="23921656")

    assert served == ["EXT_ID:23921656 AND SRC:MED"]
    assert record is not None
    assert (record.doi, record.first_author, record.year) == (
        "10.1213/ane.0b013e318299a650",
        "Ruetzler",
        2013,
    )
    assert "Randomized Controlled Trial" in record.publication_types
    assert not record.retracted
    assert [c.query for c in epmc.calls] == served  # recorded for replay


def test_retracted_by_publication_type(make_epmc) -> None:
    types = {"pubType": ["Retracted Publication", "Journal Article"]}
    epmc, served = make_epmc([epmc_result(pubTypeList=types)])

    record = epmc.find(pmid="9500320")

    assert record is not None and record.retracted


def test_retracted_by_retraction_notice(make_epmc) -> None:
    notice = {"commentCorrection": [{"type": "Comment in"}, {"type": "Retraction in"}]}
    epmc, served = make_epmc([epmc_result(commentCorrectionList=notice)])

    record = epmc.find(doi="10.1016/s0140-6736(97)11096-0")

    assert served == ['DOI:"10.1016/s0140-6736(97)11096-0"']
    assert record is not None and record.retracted


def test_no_match_returns_none(make_epmc) -> None:
    assert make_epmc([])[0].find(pmid="1") is None


def test_title_lookup_accepts_only_a_close_title(make_epmc) -> None:
    epmc, served = make_epmc([epmc_result()])

    assert (
        epmc.find(title="Licorice versus sugar-water gargle for postoperative sore throat") is None
    )
    assert epmc.find(title=D1_TITLE.upper()) is not None
    assert served[1].startswith('TITLE:"a randomized double blind comparison')


def test_ids_are_tried_before_the_title(make_epmc) -> None:
    epmc, served = make_epmc([epmc_result()])

    epmc.find(pmid="23921656", doi="10.1213/x", title=D1_TITLE)

    assert served == ["EXT_ID:23921656 AND SRC:MED"]


def test_replay_answers_without_network(make_epmc) -> None:
    live, _ = make_epmc([epmc_result()])
    live.find(pmid="23921656")

    replayed = EuropePMC.from_replay(live.calls)

    assert replayed.find(pmid="23921656") == live.find(pmid="23921656")
    with pytest.raises(KeyError):
        replayed.find(pmid="999")


def test_doi_in_url_stops_at_a_slug() -> None:
    url = (
        "https://journals.lww.com/anesthesia-analgesia/abstract/"
        "10.1213/ane.0b013e318299a650~a-randomized-double-blind-comparison"
    )

    assert ids_from_url(url) == {"doi": "10.1213/ane.0b013e318299a650"}


def test_title_is_tried_when_the_id_finds_nothing() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        hits = [] if request.url.params["query"].startswith("DOI") else [epmc_result()]
        body = {"hitCount": len(hits), "resultList": {"result": hits}}
        return httpx.Response(200, json=body)

    epmc = EuropePMC(httpx.Client(transport=httpx.MockTransport(handle)))

    record = epmc.find(doi="10.1213/wrong", title=D1_TITLE)

    assert record is not None and record.pmid == "23921656"
    assert [c.query.split(":")[0] for c in epmc.calls] == ["DOI", "TITLE"]


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        (D1_TITLE, D1_TITLE.upper(), True),
        (
            "An evaluation of the efficacy of licorice gargle for attenuating postoperative "
            "sore throat: a prospective, randomized, single-blind study.",
            "An Evaluation of the Efficacy of Licorice Gargle for Attenuating Postoperative "
            "Sore Throat",
            True,
        ),
        ("Licorice: a review", "Licorice: a randomized trial in children", False),  # short main
        ("Licorice gargle and sore throat", "Ketamine gargle and sore throat after surgery", False),
    ],
)
def test_titles_match(a: str, b: str, same: bool) -> None:
    assert titles_match(a, b) is same


def flaky_epmc(*failures: int | Exception) -> tuple[EuropePMC, list[str]]:
    """A client whose first requests fail (an HTTP status or a raised error), then succeed."""
    queue, served = list(failures), []

    def handle(request: httpx.Request) -> httpx.Response:
        served.append(str(request.url))
        if queue:
            failure = queue.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure)
        body = {"resultList": {"result": [epmc_result()]}}
        return httpx.Response(200, json=body)

    http = httpx.Client(transport=httpx.MockTransport(handle))
    return EuropePMC(http, retry_wait_s=0), served


@pytest.mark.parametrize("failure", [429, httpx.ReadTimeout("slow")])
def test_timeout_or_rate_limit_gets_one_retry(failure) -> None:
    epmc, served = flaky_epmc(failure)

    assert epmc.find(pmid="23921656") is not None
    assert len(served) == 2


def test_second_rate_limit_gives_no_record_and_records_the_error(caplog) -> None:
    # Live D0 (run 20261114-193249-8d8976): a Europe PMC outage failed 4 of 42 lookups and the
    # first one ended the whole run. A study without Europe PMC data is kept instead (user's
    # choice, Nov 14): it misses its ids and retraction check, and the error is logged.
    epmc, served = flaky_epmc(429, 429)

    with caplog.at_level(logging.ERROR):
        assert epmc.find(pmid="23921656") is None
    assert len(served) == 2
    (call,) = epmc.calls
    assert call.record is None and call.error is not None and "429" in call.error
    assert "Europe PMC lookup failed" in caplog.text


def test_other_http_errors_are_not_retried() -> None:
    epmc, served = flaky_epmc(500)

    assert epmc.find(pmid="23921656") is None
    assert len(served) == 1
    assert epmc.calls[0].error is not None and "500" in epmc.calls[0].error


def test_rate_limit_then_timeout_is_not_retried_twice() -> None:
    epmc, served = flaky_epmc(429, httpx.ReadTimeout("slow"))

    assert epmc.find(pmid="23921656") is None
    assert len(served) == 2


def test_failed_lookup_falls_back_to_the_title() -> None:
    epmc, served = flaky_epmc(500)

    assert epmc.find(pmid="23921656", title=D1_TITLE) is not None
    assert len(served) == 2


def test_failed_lookup_replays_as_no_record() -> None:
    live, _ = flaky_epmc(500)
    live.find(pmid="23921656")

    replayed = EuropePMC.from_replay(live.calls)

    assert replayed.find(pmid="23921656") is None
