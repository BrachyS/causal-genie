"""Frame a question as a `CausalQuery`: PECO plus the rest of the estimand.

If the model finds the question ambiguous, the user gets exactly one clarifying question; the
answer is folded into a second framing. With no user to ask (batch, eval), the first framing
stands and its ambiguities stay on record for the report.
"""

import logging
from collections.abc import Callable

from causal_genie.agent.state import CausalQuery
from causal_genie.llm.client import LLMClient

log = logging.getLogger(__name__)


def frame(question: str, llm: LLMClient, ask: Callable[[str], str] | None) -> CausalQuery:
    """Frame `question` as a `CausalQuery`, asking `ask` at most one clarifying question.

    With `ask=None` the first framing is kept, ambiguities and all.
    """
    query = llm.complete("small", "frame", question, CausalQuery)
    if not query.ambiguities:
        return query
    if ask is None:
        log.warning("question is ambiguous, no user to ask: %s", query.ambiguities)
        return query
    clarifying = query.ambiguities[0]
    answer = ask(clarifying)
    return llm.complete(
        "small",
        "frame",
        f"Question: {question}\nClarifying question: {clarifying}\nUser's answer: {answer}",
        CausalQuery,
    )
