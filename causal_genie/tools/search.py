"""Tavily web search and page extraction, recorded so every run can be replayed without keys.

Search results are untrusted web content: they are stored and shown as data, never followed
as instructions.
"""

import time

from tavily import TavilyClient

from causal_genie.agent.state import ExtractCall, SearchCall, SearchHit
from causal_genie.log import log_step


class Searcher:
    """Runs Tavily searches and extracts and records each one, so runs replay without keys.

    With `tavily=None` it answers only from `replay`.
    """

    def __init__(
        self,
        tavily: TavilyClient | None,
        max_results: int = 5,
        replay: list[SearchCall] | None = None,
        replay_extracts: list[ExtractCall] | None = None,
    ) -> None:
        self.tavily = tavily
        self.max_results = max_results
        self.replay = {c.query: c for c in replay or []}
        self.replay_extracts = {c.url: c for c in replay_extracts or []}
        self.calls: list[SearchCall] = []
        self.extracts: list[ExtractCall] = []

    @classmethod
    def from_replay(
        cls, calls: list[SearchCall], extracts: list[ExtractCall] | None = None
    ) -> "Searcher":
        """A searcher that answers only from a saved run's searches and extracts; no keys."""
        return cls(tavily=None, replay=calls, replay_extracts=extracts)

    def search(self, query: str) -> list[SearchHit]:
        """Hits for `query`, live or from the recording; a replay miss is an error."""
        with log_step("search", query=query) as rec:
            if self.tavily is None:
                if query not in self.replay:
                    raise KeyError(f"no recorded search for {query!r}")
                call = self.replay[query]
            else:
                start = time.perf_counter()
                found = self.tavily.search(
                    query, search_depth="advanced", max_results=self.max_results
                )
                hits = [
                    SearchHit(
                        url=r["url"], title=r["title"], content=r["content"], score=r["score"]
                    )
                    for r in found["results"]
                ]
                call = SearchCall(query=query, hits=hits, latency_s=time.perf_counter() - start)
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
