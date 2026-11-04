"""Tavily web search, recorded so every run can be replayed without keys.

Search results are untrusted web content: they are stored and shown as data, never followed
as instructions.
"""

import time

from tavily import TavilyClient

from causal_genie.agent.state import CausalQuery, SearchCall, SearchHit
from causal_genie.log import log_step


class Searcher:
    """Runs Tavily searches and records each one, so runs replay without keys.

    With `tavily=None` it answers only from `replay`.
    """

    def __init__(
        self,
        tavily: TavilyClient | None,
        max_results: int = 5,
        replay: list[SearchCall] | None = None,
    ) -> None:
        self.tavily = tavily
        self.max_results = max_results
        self.replay = {c.query: c for c in replay or []}
        self.calls: list[SearchCall] = []

    @classmethod
    def from_replay(cls, calls: list[SearchCall]) -> "Searcher":
        """A searcher that answers only from a saved run's searches; no keys or network."""
        return cls(tavily=None, replay=calls)

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


def thin_slice_queries(query: CausalQuery) -> list[str]:
    """Fixed queries for the T5 thin slice; the model-planned research loop (T6) replaces them."""
    exposure, outcome = query.exposure.definition, query.outcome.definition
    return [
        f"{exposure} vs {query.comparator} {outcome} randomized controlled trial",
        f"{exposure} {outcome} systematic review",
        f"{exposure} {outcome} trial dataset csv",
    ]
