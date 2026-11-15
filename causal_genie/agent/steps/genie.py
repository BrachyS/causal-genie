"""Genie mode (PRD §5.8): an optional playful aside, off unless `GENIE_MODE` is set.

The large model (reasoning off) judges whether the topic is sensitive and drafts the aside;
the harness decides whether it is shown. Any verdict may get one (decided Nov 14: the aside
carries no facts or numbers, so the verdict doesn't make it riskier), but never on a
sensitive topic, never with a number that isn't in the question, and it always ends with
the model's useful next step. The model's `whimsical` label is only logged.
"""

import logging

from causal_genie.agent.state import GenieJudgment, RunState
from causal_genie.agent.steps.report import VERDICT_LABELS, free_numbers, numbers
from causal_genie.llm.client import LLMClient

log = logging.getLogger(__name__)


def genie_aside(state: RunState, llm: LLMClient) -> str | None:
    """The aside to show, or None when the gate forbids one (reason logged)."""
    assert state.verdict is not None, "the gate needs the controller's verdict"
    user = f"Question: {state.question}\nVerdict: {VERDICT_LABELS[state.verdict]}"
    # large tier: the small one labeled licorice gargling before surgery "not sensitive" and
    # wrote an aside implying it soothes (Nov 14, run 20261114-224708-06e803)
    judgment = llm.complete("large", "genie_gate", user, GenieJudgment)
    text = f"{judgment.aside.strip()} {judgment.next_step.strip()}"
    log.info("genie: sensitive=%s whimsical=%s", judgment.sensitive, judgment.whimsical)
    if judgment.sensitive:
        reason = "sensitive topic"
    elif numbers(text) - free_numbers(state):
        reason = "the aside contains a number not in the question"
    else:
        return text
    log.info("genie: no aside: %s", reason)
    return None
