"""Live smoke tests: real Token Factory and Tavily calls. Run with `uv run pytest -m live`."""

import pytest
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient

from causal_genie.agent.router import route
from causal_genie.llm.client import LLMClient, load_models

pytestmark = pytest.mark.live


class Peco(BaseModel):
    population: str
    exposure: str
    comparator: str
    outcome: str


@pytest.mark.parametrize("tier", ["small", "large"])
def test_token_factory_tier_returns_schema_valid_json(nebius: OpenAI, tier: str) -> None:
    model = load_models()[tier]
    assert model in {m.id for m in nebius.models.list().data}

    response = nebius.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return the PECO for the question as JSON."},
            {"role": "user", "content": "Does licorice gargling reduce post-op sore throat?"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Peco", "schema": Peco.model_json_schema(), "strict": True},
        },
        temperature=0,
        max_tokens=2000,
    )

    peco = Peco.model_validate_json(response.choices[0].message.content or "")
    assert "licorice" in peco.exposure.lower()


def test_tavily_search_then_extract(tavily: TavilyClient) -> None:
    found = tavily.search(
        "licorice gargle postoperative sore throat randomized trial",
        search_depth="advanced",
        max_results=5,
    )
    assert found["results"]

    url = found["results"][0]["url"]
    extracted = tavily.extract(urls=[url])
    assert extracted["results"][0]["raw_content"]


@pytest.mark.parametrize(
    ("question", "kind"),
    [
        ("Does licorice gargling reduce post-op sore throat?", "causal"),
        ("How many babies are born in the US each day?", "not_causal"),
    ],
)
def test_router_with_real_small_model(nebius: OpenAI, question: str, kind: str) -> None:
    llm = LLMClient(openai=nebius, models=load_models())

    r = route(question, llm, cached={}, gallery={})

    assert r.kind == kind
    assert llm.calls[0].tokens_in > 0
