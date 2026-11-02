"""Token Factory client: the only way the harness talks to a model.

Every call uses a prompt file as the system message, passes inputs as user content (data,
never instructions), forces JSON matching a Pydantic schema, and is recorded as an `LLMCall`
for token counts, tracing and replay. Retries are limited to what the PRD names: one API
retry with backoff and one schema-repair retry.
"""

import logging
import os
import time
from pathlib import Path
from typing import Literal, TypeVar

import openai
import yaml
from openai import OpenAI
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel, ValidationError

from causal_genie.agent.state import LLMCall
from causal_genie.log import log_step

log = logging.getLogger(__name__)

LLM_DIR = Path(__file__).parent
PROMPTS_DIR = LLM_DIR / "prompts"
MODELS_YAML = LLM_DIR / "models.yaml"

Tier = Literal["small", "large"]
T = TypeVar("T", bound=BaseModel)
TRANSIENT_ERRORS = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


def load_models(path: Path = MODELS_YAML) -> dict[str, str]:
    """Tier → Token Factory model id, from `models.yaml`."""
    tiers = yaml.safe_load(path.read_text())["tiers"]
    return {tier: spec["model"] for tier, spec in tiers.items()}


def make_openai() -> OpenAI:
    """Token Factory client from env; SDK retries off so ours is the only one."""
    return OpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.environ["NEBIUS_BASE_URL"],
        max_retries=0,
    )


class LLMClient:
    def __init__(
        self,
        openai: OpenAI | None,
        models: dict[str, str],
        prompts_dir: Path = PROMPTS_DIR,
        retry_wait_s: float = 2.0,
        replay: list[LLMCall] | None = None,
    ) -> None:
        self.openai = openai
        self.models = models
        self.prompts_dir = prompts_dir
        self.retry_wait_s = retry_wait_s
        self.replay = {(c.prompt, c.user): c for c in replay or []}
        self.calls: list[LLMCall] = []

    @classmethod
    def from_replay(cls, calls: list[LLMCall], prompts_dir: Path = PROMPTS_DIR) -> "LLMClient":
        """A client that answers only from a saved run's calls; no keys or network needed."""
        models: dict[str, str] = {c.tier: c.model for c in calls}
        return cls(openai=None, models=models, prompts_dir=prompts_dir, replay=calls)

    def complete(self, tier: Tier, prompt: str, user: str, schema: type[T]) -> T:
        """Run prompt `prompt` on `user` content with the `tier` model; return validated output."""
        inputs = {"prompt": prompt, "tier": tier, "user": user[:200], "user_chars": len(user)}
        with log_step("llm", **inputs) as rec:
            if self.openai is None:
                call = self._replayed(prompt, user)
                output = schema.model_validate_json(call.response)
            else:
                call, output = self._live(tier, prompt, user, schema)
            self.calls.append(call)
            rec.update(model=call.model, tokens_in=call.tokens_in, tokens_out=call.tokens_out)
            rec["output"] = output.model_dump()
            return output

    def _replayed(self, prompt: str, user: str) -> LLMCall:
        if (prompt, user) not in self.replay:
            raise KeyError(f"no recorded call for prompt {prompt!r} with this input")
        return self.replay[(prompt, user)]

    def _live(self, tier: Tier, prompt: str, user: str, schema: type[T]) -> tuple[LLMCall, T]:
        model = self.models[tier]
        messages = [
            {"role": "system", "content": (self.prompts_dir / f"{prompt}.md").read_text()},
            {"role": "user", "content": user},
        ]
        response_format: ResponseFormatJSONSchema = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }
        start = time.perf_counter()

        text, tokens_in, tokens_out = self._ask(model, messages, response_format)
        try:
            output = schema.model_validate_json(text)
        except ValidationError as e:
            log.warning("prompt=%s output failed schema, repairing: %s", prompt, e)
            messages += [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": f"Your output failed validation:\n{e}\n"
                    "Return corrected JSON only, matching the schema.",
                },
            ]
            text, more_in, more_out = self._ask(model, messages, response_format)
            tokens_in, tokens_out = tokens_in + more_in, tokens_out + more_out
            output = schema.model_validate_json(text)

        call = LLMCall(
            prompt=prompt,
            tier=tier,
            model=model,
            user=user,
            response=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=time.perf_counter() - start,
        )
        return call, output

    def _ask(
        self, model: str, messages: list, response_format: ResponseFormatJSONSchema
    ) -> tuple[str, int, int]:
        """One completion → (text, tokens in, tokens out); one retry on transient API errors."""
        assert self.openai is not None
        try:
            response = self._create(model, messages, response_format)
        except TRANSIENT_ERRORS as e:
            log.warning("model=%s transient API error, retrying: %r", model, e)
            time.sleep(self.retry_wait_s)
            response = self._create(model, messages, response_format)
        usage = response.usage
        tokens = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
        return response.choices[0].message.content or "", *tokens

    def _create(self, model: str, messages: list, response_format: ResponseFormatJSONSchema):
        assert self.openai is not None
        return self.openai.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=0,
            max_tokens=4000,
        )
