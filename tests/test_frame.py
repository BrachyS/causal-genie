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


WHEN = "Sore throat at which time after surgery?"
WHO = "Which adults: any surgery, or only with a breathing tube?"
SCALE = "Any sore throat, or its severity?"


def test_up_to_three_ambiguities_are_asked_together_then_one_reframe(
    make_llm, sample_state
) -> None:
    fourth = "Licorice gargle or lozenge?"
    llm, fake = make_llm(
        [framed(sample_state, [WHEN, WHO, SCALE, fourth]), framed(sample_state, [])]
    )
    asked: list[str] = []

    def ask(q: str) -> str:
        asked.append(q)
        return f"answer {len(asked)}"

    query = frame(QUESTION, llm, ask=ask)

    assert asked == [WHEN, WHO, SCALE]  # most important first; the fourth is never asked
    assert len(fake.requests) == 2  # one reframe with all answers
    second = fake.requests[1]["messages"][1]["content"]
    assert QUESTION in second
    for q, a in zip(asked, ["answer 1", "answer 2", "answer 3"], strict=True):
        assert f"Q: {q}\nA: {a}" in second
    assert fourth not in second
    assert query.ambiguities == []


def test_blank_answer_keeps_the_default(make_llm, sample_state) -> None:
    llm, fake = make_llm([framed(sample_state, [WHEN, WHO]), framed(sample_state, [])])
    answers = iter(["30 minutes", "  "])

    frame(QUESTION, llm, ask=lambda q: next(answers))

    second = fake.requests[1]["messages"][1]["content"]
    assert f"Q: {WHEN}\nA: 30 minutes" in second
    assert f"Q: {WHO}\nA: (no answer: keep your default)" in second


def test_answered_ambiguity_repeated_by_the_reframe_is_dropped_new_one_kept(
    make_llm, sample_state, step_logs
) -> None:
    reworded = "At which time after surgery is sore throat measured?"
    new = "Should patients who were not extubated be counted?"
    llm, _ = make_llm([framed(sample_state, [WHEN]), framed(sample_state, [reworded, new])])

    query = frame(QUESTION, llm, ask=lambda q: "30 minutes")

    assert query.ambiguities == [new]
    assert reworded in step_logs.text  # dropped, and logged


def test_ambiguity_without_a_user_to_ask_keeps_first_framing(make_llm, sample_state) -> None:
    llm, fake = make_llm([framed(sample_state, ["Which time point?"])])

    query = frame(QUESTION, llm, ask=None)

    assert query.ambiguities == ["Which time point?"]
    assert len(fake.requests) == 1
