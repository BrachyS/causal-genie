"""Tavily web search and page extraction, recorded so every run can be replayed without keys.

Search results are untrusted web content: they are stored and shown as data, never followed
as instructions.

Research searches ask Tavily to *prefer* scholarly sources (PRD §5.3), never to restrict to
them: news, blogs and critiques still come back as leads and for the disproving search.
"""

import time
from typing import Literal

from tavily import TavilyClient

from causal_genie.agent.state import ExtractCall, SearchCall, SearchHit
from causal_genie.log import log_step

# PRD §5.3 preferred sources: PubMed, Europe PMC, Cochrane, NIH/CDC, NBER, medRxiv.
RESEARCH_DOMAINS: tuple[str, ...] = (
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "europepmc.org",
    "cochranelibrary.com",
    "nih.gov",
    "cdc.gov",
    "nber.org",
    "medrxiv.org",
)


class Searcher:
    """Runs Tavily searches and extracts and records each one, so runs replay without keys.

    With `tavily=None` it answers only from `replay`.
    """

    def __init__(
        self,
        tavily: TavilyClient | None,
        max_results: int = 10,
        replay: list[SearchCall] | None = None,
        replay_extracts: list[ExtractCall] | None = None,
    ) -> None:
        self.tavily = tavily
        self.max_results = max_results
        self.replay = {(c.query, tuple(c.domains)): c for c in replay or []}
        self.replay_extracts = {c.url: c for c in replay_extracts or []}
        self.calls: list[SearchCall] = []
        self.extracts: list[ExtractCall] = []

    @classmethod
    def from_replay(
        cls, calls: list[SearchCall], extracts: list[ExtractCall] | None = None
    ) -> "Searcher":
        """A searcher that answers only from a saved run's searches and extracts; no keys."""
        return cls(tavily=None, replay=calls, replay_extracts=extracts)

    def search(
        self,
        query: str,
        domains: tuple[str, ...] = (),
        mode: Literal["prefer", "restrict"] = "prefer",
    ) -> list[SearchHit]:
        """Hits for `query`, preferring (research) or restricted to (data) `domains` if given;
        live or from the recording (matched on query and domains), a replay miss is an error."""
        with log_step("search", query=query, domains=domains, mode=mode) as rec:
            if self.tavily is None:
                if (query, domains) not in self.replay:
                    raise KeyError(f"no recorded search for {query!r} with domains {domains}")
                call = self.replay[(query, domains)]
            else:
                start = time.perf_counter()
                if domains:
                    found = self.tavily.search(
                        query,
                        search_depth="advanced",
                        max_results=self.max_results,
                        include_domains=list(domains),
                        include_domains_mode=mode,
                    )
                else:
                    found = self.tavily.search(
                        query, search_depth="advanced", max_results=self.max_results
                    )
                hits = [
                    SearchHit(
                        url=r["url"], title=r["title"], content=r["content"], score=r["score"]
                    )
                    for r in found["results"]
                ]
                call = SearchCall(
                    query=query,
                    hits=hits,
                    latency_s=time.perf_counter() - start,
                    domains=list(domains),
                    mode=mode,
                )
            self.calls.append(call)
            rec["urls"] = [h.url for h in call.hits]
            return call.hits

    def extract(self, url: str) -> ExtractCall:
        """Full text of `url`, live or from the recording; a page Tavily can't fetch is
        recorded with its error (it is evidence of what was tried), a replay miss raises."""
        with log_step("extract", url=url) as rec:
            if self.tavily is None:
                if url not in self.replay_extracts:
                    raise KeyError(f"no recorded extract for {url!r}")
                call = self.replay_extracts[url]
            else:
                start = time.perf_counter()
                found = self.tavily.extract(urls=[url])
                ok, failed = found["results"], found.get("failed_results", [])
                call = ExtractCall(
                    url=url,
                    content=ok[0]["raw_content"] if ok else None,
                    error=None if ok else failed[0]["error"],
                    latency_s=time.perf_counter() - start,
                )
            self.extracts.append(call)
            rec.update(chars=len(call.content or ""), error=call.error)
            return call
