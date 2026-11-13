"""Frame a question as a `CausalQuery`: PECO plus the rest of the estimand.

If the model finds the question ambiguous, the user is asked up to 3 clarifying questions in
one turn (most important first; a blank answer keeps the model's default), and all answers go
into one second framing. No further rounds: at most one user turn and two frame calls. The
second framing may ask again what the user just answered; those repeats are dropped, so the
report never lists an answered question as unresolved. With no user to ask (batch, eval), the
first framing stands and its ambiguities stay on record for the report.
"""

import logging
from collections.abc import Callable

from rapidfuzz import fuzz, utils

from causal_genie.agent.state import CausalQuery
from causal_genie.llm.client import LLMClient

log = logging.getLogger(__name__)

MAX_QUESTIONS = 3
# rapidfuzz token_set_ratio; at or above this, a remaining ambiguity repeats an asked question
SAME_QUESTION = 85
NO_ANSWER = "(no answer: keep your default)"


def frame(question: str, llm: LLMClient, ask: Callable[[str], str] | None) -> CausalQuery:
    """Frame `question` as a `CausalQuery`, asking `ask` up to 3 clarifying questions in one
    turn. With `ask=None` the first framing is kept, ambiguities and all."""
    query = llm.complete("small", "frame", question, CausalQuery)
    if not query.ambiguities:
        return query
    if ask is None:
        log.warning("question is ambiguous, no user to ask: %s", query.ambiguities)
        return query
    asked = query.ambiguities[:MAX_QUESTIONS]
    answers = {q: ask(q).strip() or NO_ANSWER for q in asked}
    pairs = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in answers.items())
    reframed = llm.complete(
        "small",
        "frame",
        f"Question: {question}\n\nClarifying questions and the user's answers:\n\n{pairs}",
        CausalQuery,
    )
    remaining = []
    for ambiguity in reframed.ambiguities:
        if any(
            fuzz.token_set_ratio(ambiguity, q, processor=utils.default_process) >= SAME_QUESTION
            for q in asked
        ):
            log.info("dropped ambiguity %r: the user already answered it", ambiguity)
        else:
            remaining.append(ambiguity)
    return reframed.model_copy(update={"ambiguities": remaining})
