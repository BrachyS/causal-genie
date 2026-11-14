import httpx2
import openai
import pytest
from pydantic import BaseModel

from causal_genie.llm.client import (
    LLMClient,
    ModelOutputError,
    load_max_tokens,
    load_models,
    load_no_thinking,
    load_prompt_max_tokens,
)


class Answer(BaseModel):
    value: int


REQUEST = httpx2.Request("POST", "https://test/v1/chat/completions")


def test_complete_returns_validated_output_and_records_call(make_llm) -> None:
    llm, fake = make_llm(['{"value": 7}'])

    assert llm.complete("small", "echo", "what is 3+4?", Answer) == Answer(value=7)

    (request,) = fake.requests
    assert request["model"] == "test/small"
    assert request["messages"][0] == {"role": "system", "content": "Return the answer as JSON."}
    assert request["messages"][1] == {"role": "user", "content": "what is 3+4?"}
    assert request["response_format"]["type"] == "json_schema"
    (call,) = llm.calls
    assert (call.prompt_name, call.tier, call.model) == ("echo", "small", "test/small")
    assert (call.response, call.tokens_in, call.tokens_out) == ('{"value": 7}', 10, 5)


def test_invalid_output_gets_one_repair_retry(make_llm) -> None:
    llm, fake = make_llm(['{"value": "seven"}', '{"value": 7}'])

    assert llm.complete("small", "echo", "what is 3+4?", Answer) == Answer(value=7)

    repair = fake.requests[1]["messages"]
    assert repair[2] == {"role": "assistant", "content": '{"value": "seven"}'}
    assert "value" in repair[3]["content"]  # the validation error is shown to the model
    assert llm.calls[0].tokens_in == 20  # both attempts are counted


def test_second_invalid_output_raises_naming_the_prompt(make_llm) -> None:
    llm, _ = make_llm(["not json", '{"wrong": 1}'])

    with pytest.raises(ModelOutputError, match="prompt 'echo'(.|\n)*value"):
        llm.complete("small", "echo", "what is 3+4?", Answer)


def test_failed_repair_is_recorded_with_its_tokens_and_error(make_llm) -> None:
    llm, _ = make_llm(["not json", '{"wrong": 1}'])

    with pytest.raises(ModelOutputError):
        llm.complete("small", "echo", "what is 3+4?", Answer)

    (call,) = llm.calls
    assert call.response == '{"wrong": 1}'
    assert (call.tokens_in, call.tokens_out) == (20, 10)  # both attempts are counted
    assert call.error is not None and "value" in call.error


def test_replay_of_a_failed_call_raises_again(make_llm, prompts_dir) -> None:
    live, _ = make_llm(["not json", '{"wrong": 1}'])
    with pytest.raises(ModelOutputError):
        live.complete("small", "echo", "what is 3+4?", Answer)

    replay = LLMClient.from_replay(live.calls, prompts_dir=prompts_dir)

    with pytest.raises(ModelOutputError):
        replay.complete("small", "echo", "what is 3+4?", Answer)
    assert replay.calls == live.calls


def test_transient_api_error_gets_one_retry(make_llm) -> None:
    llm, fake = make_llm([openai.APITimeoutError(request=REQUEST), '{"value": 7}'])

    assert llm.complete("large", "echo", "q", Answer) == Answer(value=7)
    assert len(fake.requests) == 2


def test_second_api_error_raises(make_llm) -> None:
    llm, _ = make_llm([openai.APITimeoutError(request=REQUEST)] * 2)

    with pytest.raises(openai.APITimeoutError):
        llm.complete("small", "echo", "q", Answer)


def test_non_transient_api_error_is_not_retried(make_llm) -> None:
    bad_request = openai.BadRequestError(
        "bad", response=httpx2.Response(400, request=REQUEST), body=None
    )
    llm, fake = make_llm([bad_request, '{"value": 7}'])

    with pytest.raises(openai.BadRequestError):
        llm.complete("small", "echo", "q", Answer)
    assert len(fake.requests) == 1


def test_replay_returns_recorded_output_without_api(make_llm, prompts_dir) -> None:
    live, _ = make_llm(['{"value": 7}'])
    live.complete("small", "echo", "what is 3+4?", Answer)

    replay = LLMClient.from_replay(live.calls, prompts_dir=prompts_dir)

    assert replay.complete("small", "echo", "what is 3+4?", Answer) == Answer(value=7)
    with pytest.raises(KeyError, match="no recorded call"):
        replay.complete("small", "echo", "a new question", Answer)


def test_load_models_maps_tier_to_model_id() -> None:
    models = load_models()

    assert set(models) == {"small", "large"}
    assert all("/" in model for model in models.values())


def test_each_tier_uses_its_own_token_cap(make_llm) -> None:
    llm, fake = make_llm(['{"value": 7}', '{"value": 7}'])

    llm.complete("small", "echo", "q", Answer)
    llm.complete("large", "echo", "q", Answer)

    caps = load_max_tokens()
    assert [r["max_tokens"] for r in fake.requests] == [caps["small"], caps["large"]]


def test_large_tier_cap_leaves_room_for_reasoning() -> None:
    # Live D4 (run 20261113-142011-8bccfe): DeepSeek spent all 4000 tokens reasoning on the DAG
    # prompt and wrote no JSON, twice. A one-line question alone used ~2000 reasoning tokens.
    assert load_max_tokens()["large"] >= 16000


def test_finish_reason_of_each_attempt_is_recorded(make_llm) -> None:
    llm, _ = make_llm([("", "length"), '{"value": 7}'])

    assert llm.complete("large", "echo", "q", Answer) == Answer(value=7)

    assert llm.calls[0].finish_reasons == ["length", "stop"]


def test_output_cut_by_the_token_cap_is_logged(make_llm, caplog) -> None:
    llm, _ = make_llm([("", "length"), '{"value": 7}'])

    llm.complete("large", "echo", "q", Answer)

    assert any("finish_reason=length" in r.getMessage() for r in caplog.records)


NO_THINKING = {"chat_template_kwargs": {"thinking": False}}


def test_report_prompt_turns_reasoning_off_and_other_prompts_keep_it(make_llm) -> None:
    # Live D0 (run 20261114-154259-6bb1d2): the report call spent all 16,000 tokens reasoning
    # and wrote nothing. A probe (Nov 14) showed only this setting stops DeepSeek-V4-Pro-0813
    # reasoning on Token Factory; `reasoning_effort` and `thinking.type` are ignored.
    llm, fake = make_llm(['{"value": 7}', '{"value": 7}'])

    llm.complete("large", "report", "q", Answer)
    llm.complete("large", "dag", "q", Answer)

    report, dag = fake.requests
    assert report["extra_body"] == NO_THINKING
    assert dag["extra_body"] is None


def test_no_thinking_prompts_come_from_models_yaml() -> None:
    assert load_no_thinking() == {"report"}


def test_extract_prompt_has_its_own_lower_cap(make_llm) -> None:
    # Live D0: extract outputs were 263 tokens (median), 651 at most, except one runaway that
    # hit the small tier's 4,000 cap on a 3,048-character snippet.
    llm, fake = make_llm(['{"value": 7}', '{"value": 7}'])

    llm.complete("small", "extract", "q", Answer)
    llm.complete("small", "echo", "q", Answer)

    extract, echo = fake.requests
    assert extract["max_tokens"] == load_prompt_max_tokens()["extract"] == 1500
    assert echo["max_tokens"] == load_max_tokens()["small"]
