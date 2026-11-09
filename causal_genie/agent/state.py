"""Shared data contracts (PRD §5.5) and the per-run state that makes every run replayable.

Model outputs are validated against these schemas; unknown fields are rejected so a model
cannot slip in fields (e.g. a verdict) that only the controller may set.
"""

from pathlib import Path
from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


Design = Literal[
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
Strategy = Literal["randomization", "adjustment", "DiD", "IV", "RDD", "other", "unclear"]
Direction = Literal["+", "-", "null", "mixed", "unclear"]
# Why a source was left out. The model picks the first five; the harness sets `retracted`.
ExclusionReason = Literal[
    "off_topic",
    "wrong_population",
    "wrong_exposure",
    "wrong_outcome",
    "not_a_study",
    "retracted",
]
PecoMatch = Literal["direct", "indirect"]
# What a review or meta-analysis pools; n/a for any other design.
Pooled = Literal["trials", "observational", "mixed", "n/a"]
# Study fields the extraction model must back with a verbatim quote.
QuotedField = Literal["design", "population", "n", "effect", "adjusted_for", "data_link"]


class Quote(Schema):
    field: QuotedField
    text: str  # copied verbatim from the source text


class QuoteFix(Schema):
    field: QuotedField
    text: str | None  # a verbatim passage, or None if the source has none for this field


class QuoteFixes(Schema):
    """The model's second try at quotes the harness could not find in the source."""

    fixes: list[QuoteFix]


class Extraction(Schema):
    """The model's reading of one source (one call per source). The harness turns it into a
    `Study`, adding what the model must not decide: id, URL, DOI/PMID and retraction status."""

    # paper = the study's own report (article, abstract, preprint, registry entry);
    # secondary = news, blog, Wikipedia or another page describing someone else's study
    source_kind: Literal["paper", "secondary"]
    included: bool
    exclusion_reason: ExclusionReason | None
    # direct = same population and exposure as the question; indirect = a close one
    peco_match: PecoMatch
    # For a secondary source, these describe the study it reports on.
    title: str
    first_author: str | None  # surname only
    year: int | None
    design: Design
    pooled: Pooled
    strategy: Strategy
    population: str
    n: int | None
    effect: str | None
    direction: Direction
    adjusted_for: list[str]
    data_link: str | None
    quotes: list[Quote]

    @model_validator(mode="after")
    def reason_matches_decision(self) -> Self:
        """An excluded source needs a reason; an included one must not have one."""
        if self.included == (self.exclusion_reason is not None):
            raise ValueError("set exclusion_reason if and only if included is false")
        return self


class Study(Schema):
    id: str
    url: str
    doi: str | None
    pmid: str | None
    title: str
    first_author: str | None  # surname; used to dedupe
    year: int | None
    design: Design
    pooled: Pooled  # a review counts at the level of what it pools
    strategy: Strategy
    included: bool
    exclusion_reason: ExclusionReason | None
    peco_match: PecoMatch  # indirect evidence lowers certainty
    population: str
    n: int | None
    effect: str | None
    direction: Direction
    adjusted_for: list[str]  # feeds DAG confounders
    data_link: str | None  # feeds paper → data
    retracted: bool
    secondary_only: bool
    quotes: dict[str, str]  # field → verbatim passage
    publication_types: list[str]  # from Europe PMC; empty if not found there
    full_text: bool  # the extraction read the full page, not just the search snippet
    also_at: list[str]  # other URLs reporting the same study, merged by dedupe


# The kinds of evidence round 1 must search for (PRD §5.3); lets the controller check coverage.
QueryGroup = Literal[
    "best_answer",
    "primary_studies",
    "mechanism",
    "confounders",
    "evidence_against",
    "data_availability",
]


class Query(Schema):
    text: str
    tool: Literal["search", "extract"]  # search = Tavily search; extract = Tavily extract (URL)
    group: QueryGroup
    why: str  # logged in the research log; not used for control

    @model_validator(mode="after")
    def extract_needs_url(self) -> Self:
        """Tavily extract fetches a page, so its text must be a URL the model already found."""
        if self.tool == "extract" and not self.text.startswith(("https://", "http://")):
            raise ValueError(f"extract query must be a URL, got {self.text!r}")
        return self


class ResearchPlan(Schema):
    queries: list[Query]
    done: bool  # a proposal; the controller makes the final call
    done_reason: str


class FirstRoundPlan(ResearchPlan):
    """Round 1 must cover every query group in at most 12 queries.

    A plan that breaks this fails validation, so the client's repair retry and then the
    preset fallback plan handle it like any other schema failure.
    """

    @model_validator(mode="after")
    def covers_every_group(self) -> Self:
        """Reject a round-1 plan that skips a group (e.g. evidence against) or is too long."""
        missing = [g for g in get_args(QueryGroup) if g not in {q.group for q in self.queries}]
        if missing:
            raise ValueError(f"round 1 must cover every group; missing: {', '.join(missing)}")
        if len(self.queries) > 12:
            raise ValueError(f"round 1 allows at most 12 queries, got {len(self.queries)}")
        return self


class FollowUpPlan(ResearchPlan):
    """Later rounds target gaps with at most 6 queries, to bound Tavily credits."""

    @model_validator(mode="after")
    def at_most_six(self) -> Self:
        """Reject an over-long follow-up plan."""
        if len(self.queries) > 6:
            raise ValueError(f"later rounds allow at most 6 queries, got {len(self.queries)}")
        return self


class ResearchRound(Schema):
    """One research round as run: the plan and whether the model or the fallback made it."""

    number: int  # 1-based
    plan: ResearchPlan
    planned_by: Literal["model", "fallback"]
    new_studies: int = 0  # included studies this round added, after dedupe
    missing: list[str] = Field(default_factory=list)  # coverage still missing after the round
    stop_reason: str | None = None  # why the controller stopped after this round, if it did


CertaintyLevel = Literal["very_low", "low", "moderate", "high"]


class Certainty(Schema):
    """Rule-based certainty label (PRD §5.4), set by the controller, never by a model."""

    level: CertaintyLevel
    reasons: list[str]  # the starting level and each step down, in words


class EvidenceFlow(Schema):
    """PRISMA-style counts: sources found → studies after dedupe → included."""

    found: int
    after_dedupe: int
    included: int
    excluded: dict[str, int]  # exclusion reason → count


ClaimKind = Literal[
    "edge",
    "confounder",
    "effect_estimate",
    "null_result",
    "mechanism",
    "data_availability",
    "study_detail",  # design, population or sample size of a study
]


class Claim(Schema):
    id: str
    text: str
    kind: ClaimKind
    study_id: str | None
    quote: str | None
    result_id: str | None
    status: Literal["documented", "assumed", "inferred", "unresolved"]
    grounding: Literal["quote_found", "quote_not_found", "n/a"]
    support: Literal["supports", "partial", "contradicts", "irrelevant", "unchecked"]


Support = Literal["supports", "partial", "contradicts", "irrelevant"]


class SupportJudgment(Schema):
    claim_id: str
    support: Support


class SupportCheck(Schema):
    """The model's verdict on whether each quote backs its claim (one call per study)."""

    judgments: list[SupportJudgment]


class DagEdge(Schema):
    source: str
    target: str
    claim_ids: list[str]  # empty → "assumed"

    @property
    def assumed(self) -> bool:
        """No ledger claim backs this edge; the report must say it is an assumption."""
        return not self.claim_ids


class DagProposal(Schema):
    """The model's proposed causal graph; the harness checks every cited claim id."""

    edges: list[DagEdge]


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
    # domains Tavily was asked to prefer; empty = no filter (and for runs saved before this)
    domains: list[str] = Field(default_factory=list)


class ExtractCall(Schema):
    """One Tavily extract (full page text) as recorded for tracing and replay."""

    url: str
    content: str | None  # page text: data to read, never instructions; None if it failed
    error: str | None  # Tavily's reason when the page could not be fetched
    latency_s: float


class PaperRecord(Schema):
    """Bibliographic facts from Europe PMC; the harness trusts these over the model's reading."""

    pmid: str | None
    doi: str | None
    title: str
    first_author: str | None
    year: int | None
    publication_types: list[str]
    retracted: bool


class LookupCall(Schema):
    """One Europe PMC search as recorded for tracing and replay."""

    query: str  # Europe PMC query syntax, e.g. EXT_ID:23921656 AND SRC:MED
    record: PaperRecord | None  # None: no match
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
    rounds: list[ResearchRound] = Field(default_factory=list)
    studies: list[Study] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)  # the claim ledger
    certainty: Certainty | None = None  # set by the controller after research
    edges: list[DagEdge] = Field(default_factory=list)
    datasets: list[DatasetCandidate] = Field(default_factory=list)
    results: list[Result] = Field(default_factory=list)
    searches: list[SearchCall] = Field(default_factory=list)
    extracts: list[ExtractCall] = Field(default_factory=list)
    lookups: list[LookupCall] = Field(default_factory=list)
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
