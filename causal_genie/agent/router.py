"""Route a question: replay a cached or gallery run, answer briefly if not causal, or research it.

Cache and gallery matches are decided by code; the model only judges whether the question is
causal, and the controller acts on the returned `Route`.
"""

import re

from causal_genie.agent.state import Route, RouteDecision
from causal_genie.llm.client import LLMClient


def normalize(question: str) -> str:
    """Key for exact-repeat matching: lowercase, no punctuation, so trivial edits still match."""
    return re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()


def route(question: str, llm: LLMClient, cached: dict[str, str], gallery: dict[str, str]) -> Route:
    """Pick the route for `question`; `cached` and `gallery` map past questions to run ids.

    A past run is reused only if the question is identical, ignoring case and punctuation.
    We don't use fuzzy matching: "Does sugar raise blood pressure?" and "Does salt raise blood
    pressure?" look almost the same as text but are different questions, so a fuzzy match would
    return the wrong answer.
    """
    key = normalize(question)
    for kind, runs in (("cache", cached), ("gallery", gallery)):
        by_key = {normalize(q): run_id for q, run_id in runs.items()}
        if key in by_key:
            return Route(kind=kind, run_id=by_key[key], reason="exact repeat", causal_rewrite=None)

    decision = llm.complete("small", "route", question, RouteDecision)
    if decision.causal:
        return Route(kind="causal", run_id=None, reason=decision.reason, causal_rewrite=None)
    return Route(
        kind="not_causal",
        run_id=None,
        reason=decision.reason,
        causal_rewrite=decision.causal_rewrite,
    )
