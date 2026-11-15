import json

import pytest

from causal_genie.agent.state import RunState
from causal_genie.agent.steps.genie import genie_aside
from causal_genie.agent.steps.report import VERDICT_LABELS
from causal_genie.llm.client import load_no_thinking

ASIDE = "The moon has an alibi for every busy night on the ward."
NEXT = "Next: read the studies listed above before trusting the folklore."


def judged(sensitive: bool = False, whimsical: bool = True, aside: str = ASIDE) -> str:
    """Genie-gate model output as JSON."""
    return json.dumps(
        {"sensitive": sensitive, "whimsical": whimsical, "aside": aside, "next_step": NEXT}
    )


@pytest.mark.parametrize("verdict", list(VERDICT_LABELS))
def test_any_verdict_gets_an_aside_on_a_non_sensitive_topic(
    sample_state: RunState, make_llm, verdict
) -> None:
    sample_state.verdict = verdict
    llm, _ = make_llm([judged(whimsical=False)])

    assert genie_aside(sample_state, llm) == f"{ASIDE} {NEXT}"
    # large tier: the small one labeled a clinical treatment question "not sensitive" (Nov 14)
    assert [(c.prompt_name, c.tier) for c in llm.calls] == [("genie_gate", "large")]


@pytest.mark.parametrize("verdict", ["estimated", "published_only"])
def test_never_on_a_sensitive_topic_even_if_whimsical(
    sample_state: RunState, make_llm, step_logs, verdict
) -> None:
    sample_state.verdict = verdict
    llm, _ = make_llm([judged(sensitive=True, whimsical=True)])

    assert genie_aside(sample_state, llm) is None
    assert "sensitive topic" in step_logs.text


def test_aside_with_a_number_is_dropped(sample_state: RunState, make_llm) -> None:
    sample_state.verdict = "published_only"
    llm, _ = make_llm([judged(aside="Births rise 3% under a full moon, says nobody.")])

    assert genie_aside(sample_state, llm) is None


def test_aside_may_repeat_a_number_from_the_question(sample_state: RunState, make_llm) -> None:
    sample_state.question = "Did the new onboarding flow raise 7-day activation?"
    sample_state.query = None
    sample_state.verdict = "study_plan"
    llm, _ = make_llm([judged(aside="A 7-day wait is a long time for a welcome mat.")])

    assert genie_aside(sample_state, llm) is not None


def test_genie_gate_runs_without_reasoning() -> None:
    assert "genie_gate" in load_no_thinking()


def test_model_sees_question_and_verdict_but_no_numbers_from_results(
    sample_state: RunState, make_llm
) -> None:
    sample_state.verdict = "published_only"
    llm, _ = make_llm([judged()])

    genie_aside(sample_state, llm)

    user = llm.calls[0].user_content
    assert sample_state.question in user and "📚 Published evidence only" in user
