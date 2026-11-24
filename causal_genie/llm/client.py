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


class ModelOutputError(ValueError):
    """A model's output still failed its schema after the one repair retry (PRD §5.9).

    Separate from pydantic's `ValidationError` so callers can tell a model's bad output from
    a bug in the harness's own schema use: the controller turns this into a partial report.
    """


def load_models(path: Path = MODELS_YAML) -> dict[str, str]:
    """Tier → Token Factory model id, from `models.yaml`."""
    tiers = yaml.safe_load(path.read_text())["tiers"]
    return {tier: spec["model"] for tier, spec in tiers.items()}


def load_max_tokens(path: Path = MODELS_YAML) -> dict[str, int]:
    """Tier → completion token cap, from `models.yaml`; reasoning models need room to think."""
    tiers = yaml.safe_load(path.read_text())["tiers"]
    return {tier: spec["max_tokens"] for tier, spec in tiers.items()}


def load_prompt_max_tokens(path: Path = MODELS_YAML) -> dict[str, int]:
    """Prompt name → completion token cap overriding its tier's, from `models.yaml`."""
    return yaml.safe_load(path.read_text()).get("prompt_max_tokens", {})


def load_no_thinking(path: Path = MODELS_YAML) -> set[str]:
    """Prompt names whose calls turn the model's reasoning off, from `models.yaml`."""
    return set(yaml.safe_load(path.read_text()).get("no_thinking", []))


# The only setting that stops DeepSeek-V4-Pro-0813 reasoning on Token Factory (probe);
# `reasoning_effort` and `thinking: {type: disabled}` are accepted but ignored.
NO_THINKING = {"chat_template_kwargs": {"thinking": False}}


def make_openai() -> OpenAI:
    """Token Factory client from env; SDK retries off so ours is the only one."""
    return OpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.environ["NEBIUS_BASE_URL"],
        max_retries=0,
    )


class LLMClient:
    """Calls models with a prompt file and a schema, recording every call for replay.

    With `openai=None` it answers only from `replay`, so saved runs reproduce offline.
    """

    def __init__(
        self,
        openai: OpenAI | None,
        models: dict[str, str],
        prompts_dir: Path = PROMPTS_DIR,
        retry_wait_s: float = 2.0,
        replay: list[LLMCall] | None = None,
        max_tokens: dict[str, int] | None = None,
        no_thinking: set[str] | None = None,
    ) -> None:
        self.openai = openai
        self.models = models
        self.max_tokens = max_tokens or load_max_tokens()
        self.no_thinking = load_no_thinking() if no_thinking is None else no_thinking
        self.prompt_max_tokens = load_prompt_max_tokens()
        self.prompts_dir = prompts_dir
        self.retry_wait_s = retry_wait_s
        self.replay = {(c.prompt_name, c.user_content): c for c in replay or []}
        self.calls: list[LLMCall] = []

    @classmethod
    def from_replay(cls, calls: list[LLMCall], prompts_dir: Path = PROMPTS_DIR) -> "LLMClient":
        """A client that answers only from a saved run's calls; no keys or network needed."""
        models: dict[str, str] = {c.tier: c.model for c in calls}
        return cls(openai=None, models=models, prompts_dir=prompts_dir, replay=calls)

    def complete(self, tier: Tier, prompt_name: str, user_content: str, schema: type[T]) -> T:
        """Ask the `tier` model to apply `prompts/<prompt_name>.md` to `user_content`.

        Returns an instance of `schema`: the model is forced to answer in that shape, and its
        JSON is validated into it. Raises `ModelOutputError` if it still fails after repair.
        """
        inputs = {
            "prompt_name": prompt_name,
            "tier": tier,
            "user_content": user_content[:200],
            "user_chars": len(user_content),
        }
        with log_step("llm", **inputs) as rec:
            earlier = self._answered(prompt_name, user_content)
            if self.openai is None:
                call = self._replayed(prompt_name, user_content)
            elif earlier is not None:
                log.info("prompt %s: same input as an earlier call; answer reused", prompt_name)
                call = earlier
            else:
                call = self._live(tier, prompt_name, user_content, schema)
            # Recorded before validating, so a failed call still shows its tokens and output.
            self.calls.append(call)
            rec.update(model=call.model, tokens_in=call.tokens_in, tokens_out=call.tokens_out)
            try:
                output = schema.model_validate_json(call.response)
            except ValidationError as e:
                raise ModelOutputError(
                    f"prompt {prompt_name!r}: output failed schema after repair: {e}"
                ) from e
            rec["output"] = output.model_dump()
            return output

    def _answered(self, prompt_name: str, user_content: str) -> LLMCall | None:
        """An earlier successful call in this run with the same prompt and input. Replay keeps
        one answer per (prompt, input), so a run must not record two different ones."""
        for c in self.calls:
            if (c.prompt_name, c.user_content) == (prompt_name, user_content) and not c.error:
                return c
        return None

    def _replayed(self, prompt_name: str, user_content: str) -> LLMCall:
        """The recorded call for this prompt and input; a miss is an error, not a live call."""
        if (prompt_name, user_content) not in self.replay:
            raise KeyError(f"no recorded call for prompt {prompt_name!r} with this input")
        return self.replay[(prompt_name, user_content)]

    def _live(self, tier: Tier, prompt_name: str, user_content: str, schema: type[T]) -> LLMCall:
        """Call the model, with one repair retry if its JSON fails the schema check.

        If the repair also fails, the call is returned with `error` set; `complete` then raises.
        """
        model = self.models[tier]
        messages = [
            {"role": "system", "content": (self.prompts_dir / f"{prompt_name}.md").read_text()},
            {"role": "user", "content": user_content},
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

        text, tokens_in, tokens_out, finish = self._ask(
            tier, prompt_name, messages, response_format
        )
        finish_reasons = [finish]
        error = None
        try:
            schema.model_validate_json(text)
        except ValidationError as e:
            log.warning("prompt=%s output failed schema, repairing: %s", prompt_name, e)
            messages += [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": f"Your output failed validation:\n{e}\n"
                    "Return corrected JSON only, matching the schema.",
                },
            ]
            text, more_in, more_out, finish = self._ask(
                tier, prompt_name, messages, response_format
            )
            tokens_in, tokens_out = tokens_in + more_in, tokens_out + more_out
            finish_reasons.append(finish)
            try:
                schema.model_validate_json(text)
            except ValidationError as e2:
                error = str(e2)

        call = LLMCall(
            prompt_name=prompt_name,
            tier=tier,
            model=model,
            user_content=user_content,
            response=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=time.perf_counter() - start,
            error=error,
            finish_reasons=finish_reasons,
        )
        return call

    def _ask(
        self,
        tier: Tier,
        prompt_name: str,
        messages: list,
        response_format: ResponseFormatJSONSchema,
    ) -> tuple[str, int, int, str]:
        """One completion → (text, tokens in, tokens out, finish reason); one retry on
        transient API errors. A cut-off output (`length`) is logged as a warning."""
        assert self.openai is not None
        try:
            response = self._create(tier, prompt_name, messages, response_format)
        except TRANSIENT_ERRORS as e:
            log.warning("tier=%s transient API error, retrying: %r", tier, e)
            time.sleep(self.retry_wait_s)
            response = self._create(tier, prompt_name, messages, response_format)
        usage = response.usage
        tokens = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
        choice = response.choices[0]
        level = logging.WARNING if choice.finish_reason == "length" else logging.INFO
        log.log(
            level,
            "model=%s finish_reason=%s tokens_out=%d max_tokens=%d",
            self.models[tier],
            choice.finish_reason,
            tokens[1],
            self.cap(tier, prompt_name),
        )
        return choice.message.content or "", *tokens, choice.finish_reason

    def cap(self, tier: Tier, prompt_name: str) -> int:
        """The prompt's own token cap if `models.yaml` sets one, else its tier's."""
        return self.prompt_max_tokens.get(prompt_name, self.max_tokens[tier])

    def _create(
        self,
        tier: Tier,
        prompt_name: str,
        messages: list,
        response_format: ResponseFormatJSONSchema,
    ):
        """The raw API request; deterministic (temperature 0), capped per tier or prompt, and with
        reasoning off for the prompts `models.yaml` names under `no_thinking`."""
        assert self.openai is not None
        return self.openai.chat.completions.create(
            model=self.models[tier],
            messages=messages,
            response_format=response_format,
            temperature=0,
            max_tokens=self.cap(tier, prompt_name),
            extra_body=NO_THINKING if prompt_name in self.no_thinking else None,
        )
