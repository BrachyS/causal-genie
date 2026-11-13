"""Genie mode (PRD §5.8): an optional playful aside, off unless `GENIE_MODE` is set.

The small model judges whether the topic is sensitive or whimsical and drafts the aside; the
harness decides whether it is shown. An aside is allowed only after a verdict without an
estimate (📚, ❓, 🚫) or for a whimsical question, never on a sensitive topic, never with a
number in it, and it always ends with the model's useful next step.
"""

import logging

from causal_genie.agent.state import GenieJudgment, RunState, Verdict
from causal_genie.agent.steps.report import NUMBER, VERDICT_LABELS
from causal_genie.llm.client import LLMClient

log = logging.getLogger(__name__)

NO_ESTIMATE: set[Verdict] = {"published_only", "insufficient", "not_causal"}


def genie_aside(state: RunState, llm: LLMClient) -> str | None:
    """The aside to show, or None when the gate forbids one (reason logged)."""
    assert state.verdict is not None, "the gate needs the controller's verdict"
    user = f"Question: {state.question}\nVerdict: {VERDICT_LABELS[state.verdict]}"
    judgment = llm.complete("small", "genie_gate", user, GenieJudgment)
    text = f"{judgment.aside.strip()} {judgment.next_step.strip()}"
    if judgment.sensitive:
        reason = "sensitive topic"
    elif state.verdict not in NO_ESTIMATE and not judgment.whimsical:
        reason = f"verdict {state.verdict} and the question is not whimsical"
    elif NUMBER.search(text):
        reason = "the aside contains a number"
    else:
        return text
    log.info("genie: no aside: %s", reason)
    return None
