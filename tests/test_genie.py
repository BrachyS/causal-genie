import json

import pytest

from causal_genie.agent.state import RunState
from causal_genie.agent.steps.genie import genie_aside

ASIDE = "The moon has an alibi for every busy night on the ward."
NEXT = "Next: read the studies listed above before trusting the folklore."


def judged(sensitive: bool = False, whimsical: bool = True, aside: str = ASIDE) -> str:
    """Genie-gate model output as JSON."""
    return json.dumps(
        {"sensitive": sensitive, "whimsical": whimsical, "aside": aside, "next_step": NEXT}
    )


@pytest.mark.parametrize("verdict", ["published_only", "insufficient", "not_causal"])
def test_aside_allowed_for_no_estimate_verdicts(sample_state: RunState, make_llm, verdict) -> None:
    sample_state.verdict = verdict
    llm, _ = make_llm([judged(whimsical=False)])

    assert genie_aside(sample_state, llm) == f"{ASIDE} {NEXT}"
    assert [(c.prompt_name, c.tier) for c in llm.calls] == [("genie_gate", "small")]


def test_estimated_verdict_gets_an_aside_only_when_whimsical(sample_state, make_llm) -> None:
    sample_state.verdict = "estimated"
    llm, _ = make_llm([judged(whimsical=False), judged(whimsical=True)])

    assert genie_aside(sample_state, llm) is None
    assert genie_aside(sample_state, llm) == f"{ASIDE} {NEXT}"


def test_never_on_a_sensitive_topic(sample_state: RunState, make_llm) -> None:
    sample_state.verdict = "published_only"
    llm, _ = make_llm([judged(sensitive=True)])

    assert genie_aside(sample_state, llm) is None


def test_aside_with_a_number_is_dropped(sample_state: RunState, make_llm) -> None:
    sample_state.verdict = "published_only"
    llm, _ = make_llm([judged(aside="Births rise 3% under a full moon, says nobody.")])

    assert genie_aside(sample_state, llm) is None


def test_model_sees_question_and_verdict_but_no_numbers_from_results(
    sample_state: RunState, make_llm
) -> None:
    sample_state.verdict = "published_only"
    llm, _ = make_llm([judged()])

    genie_aside(sample_state, llm)

    user = llm.calls[0].user_content
    assert sample_state.question in user and "📚 Published evidence only" in user
