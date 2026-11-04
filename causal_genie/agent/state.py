"""Shared data contracts (PRD §5.5) and the per-run state that makes every run replayable.

Model outputs are validated against these schemas; unknown fields are rejected so a model
cannot slip in fields (e.g. a verdict) that only the controller may set.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Variable(Schema):
    name: str  # short snake_case id, used as the DAG node name
    definition: str  # how it is measured or defined


class CausalQuery(Schema):
    population: str
    exposure: Variable
    comparator: str
    outcome: Variable
    time_window: str | None
    # The whole CausalQuery is the estimand; these fields complete it.
    target: Literal["ATE", "ATT"]  # average effect in everyone vs in the treated
    effect_scale: Literal["risk_difference", "risk_ratio", "odds_ratio", "mean_difference"]
    # ITT vs per-protocol; n/a if observational
    deviation_strategy: Literal["as_assigned", "as_received", "n/a"]
    ambiguities: list[str]  # non-empty → one clarifying question


class Study(Schema):
    id: str
    url: str
    doi: str | None
    pmid: str | None
    title: str
    year: int | None
    design: Literal[
        "meta_analysis",
        "systematic_review",
        "RCT",
        "natural_experiment",
        "cohort",
        "case_control",
        "cross_sectional",
        "case_report",
        "other",
    ]
    # how the study identified its effect
    strategy: Literal["randomization", "adjustment", "DiD", "IV", "RDD", "other", "unclear"]
    included: bool
    exclusion_reason: str | None
    population: str
    n: int | None
    effect: str | None
    direction: Literal["+", "-", "null", "mixed", "unclear"]
    adjusted_for: list[str]  # feeds DAG confounders
    data_link: str | None  # feeds paper → data
    retracted: bool
    secondary_only: bool
    quotes: dict[str, str]  # field → verbatim passage


class Query(Schema):
    text: str
    tool: Literal["search", "extract"]  # search = Tavily search; extract = Tavily extract (URL)
    why: str  # logged in the research log; not used for control


class ResearchPlan(Schema):
    queries: list[Query]
    done: bool  # a proposal; the controller makes the final call
    done_reason: str


class Claim(Schema):
    id: str
    text: str
    kind: Literal[
        "edge", "confounder", "effect_estimate", "null_result", "mechanism", "data_availability"
    ]
    study_id: str | None
    quote: str | None
    result_id: str | None
    status: Literal["documented", "assumed", "inferred", "unresolved"]
    grounding: Literal["quote_found", "quote_not_found", "n/a"]
    support: Literal["supports", "partial", "contradicts", "irrelevant", "unchecked"]


class DagEdge(Schema):
    source: str
    target: str
    claim_ids: list[str]  # empty → "assumed"


class DatasetCandidate(Schema):
    url: str
    host: str
    license: str | None
    access: Literal["open", "restricted", "unknown"]
    coverage: dict[str, str | None]  # DAG node → column
    score: float | None
    rejected_reason: str | None


class Result(Schema):
    id: str
    tool: str
    tool_version: str
    inputs_hash: str
    outputs: dict


class RouteDecision(Schema):
    """The router model's only judgment: is this a causal question?"""

    causal: bool
    reason: str
    causal_rewrite: str | None  # for descriptive questions: the closest causal question


class Route(Schema):
    kind: Literal["cache", "gallery", "not_causal", "causal"]
    run_id: str | None  # run to replay for cache / gallery
    reason: str
    causal_rewrite: str | None


class SearchHit(Schema):
    url: str
    title: str
    content: str  # Tavily's snippet: data to read, never instructions
    score: float


class SearchCall(Schema):
    """One web search as recorded for tracing and replay."""

    query: str
    hits: list[SearchHit]
    latency_s: float


class Download(Schema):
    """Provenance of one downloaded file, so any estimate can be traced to exact bytes."""

    url: str
    path: str  # local file under cache/; never committed
    sha256: str
    bytes: int
    retrieved_at: str  # ISO 8601, UTC
    doi: str | None
    license: str | None


Verdict = Literal[
    "estimated",
    "estimated_strong_assumptions",
    "published_only",
    "restricted",
    "study_plan",
    "insufficient",
    "not_causal",
]


class LLMCall(Schema):
    """One model call as recorded for tracing, token counts and replay."""

    prompt_name: str
    tier: Literal["small", "large"]
    model: str
    user_content: str
    response: str  # final output text; schema-valid unless `error` is set
    tokens_in: int  # summed over the repair attempt, if any
    tokens_out: int
    latency_s: float
    error: str | None = None  # validation error left after the repair attempt


class RunState(Schema):
    run_id: str
    question: str
    route: Route | None = None
    # clarifying question → user's answer; replay reuses the answers
    clarifications: dict[str, str] = Field(default_factory=dict)
    query: CausalQuery | None = None
    plans: list[ResearchPlan] = Field(default_factory=list)
    studies: list[Study] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)  # the claim ledger
    edges: list[DagEdge] = Field(default_factory=list)
    datasets: list[DatasetCandidate] = Field(default_factory=list)
    results: list[Result] = Field(default_factory=list)
    searches: list[SearchCall] = Field(default_factory=list)
    downloads: list[Download] = Field(default_factory=list)
    verdict: Verdict | None = None  # set by the controller only
    report: str | None = None
    llm_calls: list[LLMCall] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)  # models, tools, package

    @property
    def tokens(self) -> dict[str, int]:
        """Total model tokens in and out across the run, for cost tracking."""
        return {
            "in": sum(c.tokens_in for c in self.llm_calls),
            "out": sum(c.tokens_out for c in self.llm_calls),
        }

    @property
    def llm_latency_s(self) -> float:
        """Total time spent waiting on the model across the run."""
        return sum(c.latency_s for c in self.llm_calls)

    def save(self, runs_dir: Path) -> Path:
        """Write `runs_dir/<run_id>/state.json` so the run can be inspected or replayed."""
        path = runs_dir / self.run_id / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "RunState":
        """Read a run saved by `save`, validating it against the schema."""
        return cls.model_validate_json(path.read_text())
