import httpx2
import openai
import pytest
from pydantic import BaseModel, ValidationError

from causal_genie.llm.client import LLMClient, load_models


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
    assert (call.prompt, call.tier, call.model) == ("echo", "small", "test/small")
    assert (call.response, call.tokens_in, call.tokens_out) == ('{"value": 7}', 10, 5)


def test_invalid_output_gets_one_repair_retry(make_llm) -> None:
    llm, fake = make_llm(['{"value": "seven"}', '{"value": 7}'])

    assert llm.complete("small", "echo", "what is 3+4?", Answer) == Answer(value=7)

    repair = fake.requests[1]["messages"]
    assert repair[2] == {"role": "assistant", "content": '{"value": "seven"}'}
    assert "value" in repair[3]["content"]  # the validation error is shown to the model
    assert llm.calls[0].tokens_in == 20  # both attempts are counted


def test_second_invalid_output_raises(make_llm) -> None:
    llm, _ = make_llm(["not json", '{"wrong": 1}'])

    with pytest.raises(ValidationError):
        llm.complete("small", "echo", "what is 3+4?", Answer)


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
