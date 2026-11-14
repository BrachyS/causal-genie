"""Europe PMC lookups: the bibliographic facts the harness trusts over a model's reading.

Every new study is looked up automatically (not a model choice) for its DOI, PMID, publication
types and retraction status. Lookups are recorded so runs replay without network.
"""

import logging
import re
import time

import httpx
from rapidfuzz import fuzz

from causal_genie.agent.state import LookupCall, PaperRecord
from causal_genie.log import log_step

log = logging.getLogger(__name__)

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# rapidfuzz ratio on normalized titles; below this, a title hit is a different paper
TITLE_MATCH = 90
MAIN_TITLE_MIN_CHARS = 30  # shorter main titles ("Licorice: ...") don't identify a paper

PMID_URL = re.compile(
    r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pubmed/|europepmc\.org/"
    r"(?:abstract|article)/MED/)(\d+)"
)
PMCID_URL = re.compile(r"(?:pmc\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pmc)/articles/(PMC\d+)")
DOI_URL = re.compile(r"(10\.\d{4,9}/[^\s?#~]+)")  # a "~" starts a journal's URL slug
DOI_URL_SUFFIXES = ("/full", "/abstract", "/pdf", "/epdf", ".pdf")


def ids_from_url(url: str) -> dict[str, str]:
    """PMID, PMCID or DOI that a URL names (e.g. a PubMed or doi.org link); empty if none."""
    if m := PMID_URL.search(url):
        return {"pmid": m.group(1)}
    if m := PMCID_URL.search(url):
        return {"pmcid": m.group(1)}
    if m := DOI_URL.search(url):
        doi = m.group(1).rstrip("/")
        while doi.endswith(DOI_URL_SUFFIXES):  # e.g. .../doi/full/10.1111/x/abstract
            doi = next(doi.removesuffix(s) for s in DOI_URL_SUFFIXES if doi.endswith(s))
        return {"doi": doi}
    return {}


def normalize_title(title: str) -> str:
    """Lowercase words only, so punctuation and case don't block a title match."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def titles_match(a: str, b: str) -> bool:
    """Same paper title, ignoring case, punctuation and a subtitle one source left out.

    Main titles (before the colon) are compared only when long enough to identify a paper.
    """
    if fuzz.ratio(normalize_title(a), normalize_title(b)) >= TITLE_MATCH:
        return True
    main_a, main_b = (normalize_title(t.split(":")[0]) for t in (a, b))
    long_enough = min(len(main_a), len(main_b)) >= MAIN_TITLE_MIN_CHARS
    return long_enough and fuzz.ratio(main_a, main_b) >= TITLE_MATCH


class EuropePMC:
    """Finds a paper in Europe PMC by PMID, PMCID, DOI or title, recording each search.

    With `http=None` it answers only from `replay`.
    """

    def __init__(
        self,
        http: httpx.Client | None,
        replay: list[LookupCall] | None = None,
        retry_wait_s: float = 2.0,
    ) -> None:
        self.http = http
        self.retry_wait_s = retry_wait_s
        self.replay = {c.query: c for c in replay or []}
        self.calls: list[LookupCall] = []

    @classmethod
    def from_replay(cls, calls: list[LookupCall]) -> "EuropePMC":
        """A client that answers only from a saved run's lookups; no network."""
        return cls(http=None, replay=calls)

    def find(
        self,
        pmid: str | None = None,
        pmcid: str | None = None,
        doi: str | None = None,
        title: str | None = None,
    ) -> PaperRecord | None:
        """The paper's record, trying the first identifier given, then the title if the
        identifier finds nothing (a URL can hold a wrong-looking DOI). A title hit counts only if
        the titles match closely, since a loose match would attach another paper's DOI and
        retraction status."""
        query = None
        if pmid:
            query = f"EXT_ID:{pmid} AND SRC:MED"
        elif pmcid:
            query = f"PMCID:{pmcid}"
        elif doi:
            query = f'DOI:"{doi}"'
        if query and (record := self._search(query)):
            return record
        if title:
            record = self._search(f'TITLE:"{normalize_title(title)}"')
            if record and titles_match(record.title, title):
                return record
        return None

    def _search(self, query: str) -> PaperRecord | None:
        """Top Europe PMC hit for `query`, live or from the recording; a replay miss raises."""
        with log_step("europepmc", query=query) as rec:
            if self.http is None:
                if query not in self.replay:
                    raise KeyError(f"no recorded Europe PMC lookup for {query!r}")
                call = self.replay[query]
            else:
                start = time.perf_counter()
                response = self._get(query)
                results = response.json()["resultList"]["result"]
                record = parse_record(results[0]) if results else None
                call = LookupCall(query=query, record=record, latency_s=time.perf_counter() - start)
            self.calls.append(call)
            rec["record"] = call.record.model_dump() if call.record else None
            return call.record

    def _get(self, query: str) -> httpx.Response:
        """One search request; PRD §5.9: retried once after a wait on a timeout or a 429 rate
        limit, any other HTTP error raises at once."""
        assert self.http is not None
        params = {"query": query, "format": "json", "resultType": "core", "pageSize": 1}
        try:
            first: httpx.Response | None = self.http.get(SEARCH_URL, params=params)
        except httpx.TimeoutException as e:
            log.warning("Europe PMC timeout: %r", e)
            first = None
        if first is not None and first.status_code != 429:
            return first.raise_for_status()
        log.warning("Europe PMC timeout or rate limit, retrying once: %s", query)
        time.sleep(self.retry_wait_s)
        return self.http.get(SEARCH_URL, params=params).raise_for_status()


def parse_record(result: dict) -> PaperRecord:
    """A `PaperRecord` from one Europe PMC core result; retracted if either the publication
    type or a linked retraction notice says so."""
    types = (result.get("pubTypeList") or {}).get("pubType", [])
    notices = (result.get("commentCorrectionList") or {}).get("commentCorrection", [])
    authors = (result.get("authorList") or {}).get("author", [])
    year = result.get("pubYear")
    return PaperRecord(
        pmid=result.get("pmid"),
        doi=result.get("doi"),
        title=result["title"],
        first_author=authors[0].get("lastName") if authors else None,
        year=int(year) if year else None,
        publication_types=types,
        retracted="Retracted Publication" in types
        or any(n.get("type") == "Retraction in" for n in notices),
    )
