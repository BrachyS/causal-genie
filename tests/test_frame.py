from causal_genie.agent.state import RunState
from causal_genie.agent.steps.frame import frame

QUESTION = "Does licorice gargling reduce post-op sore throat?"


def framed(sample_state: RunState, ambiguities: list[str]) -> str:
    """Frame model output for the sample estimand, with the given ambiguities."""
    assert sample_state.query is not None
    return sample_state.query.model_copy(update={"ambiguities": ambiguities}).model_dump_json()


def test_clear_question_is_framed_in_one_call(make_llm, sample_state) -> None:
    llm, fake = make_llm([framed(sample_state, [])])

    def ask(q: str) -> str:
        raise AssertionError(f"clear question should not ask {q!r}")

    query = frame(QUESTION, llm, ask=ask)

    assert query.exposure.name == "licorice_gargle"
    assert fake.requests[0]["messages"][1]["content"] == QUESTION


def test_ambiguous_question_asks_once_and_reframes(make_llm, sample_state) -> None:
    which = "Sore throat at which time after surgery?"
    llm, fake = make_llm([framed(sample_state, [which, "Which adults?"]), framed(sample_state, [])])
    asked: list[str] = []

    def ask(q: str) -> str:
        asked.append(q)
        return "30 minutes"

    query = frame(QUESTION, llm, ask=ask)

    assert asked == [which]
    second = fake.requests[1]["messages"][1]["content"]
    assert QUESTION in second and which in second and "30 minutes" in second
    assert query.ambiguities == []


def test_ambiguity_without_a_user_to_ask_keeps_first_framing(make_llm, sample_state) -> None:
    llm, fake = make_llm([framed(sample_state, ["Which time point?"])])

    query = frame(QUESTION, llm, ask=None)

    assert query.ambiguities == ["Which time point?"]
    assert len(fake.requests) == 1
