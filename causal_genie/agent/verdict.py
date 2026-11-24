"""Pick the run's verdict (PRD §5.2) from what the run found. Rules only, never a model.

The verdict reads the run state after every step has run: an estimate beats a study plan,
a study plan beats "no data", and "no data" splits into restricted, common cause,
published-only or insufficient by what research and data discovery found.

"Common cause" is for a link nobody studies because it is spurious (ice cream sales and
shark bites): no trial or observational study of this exposure and outcome is included, and
the DAG has a node with checked citations into both. Without this, such a question could only
be "insufficient", although the sources do explain the link.
"""

import logging
import re

import networkx as nx

from causal_genie.agent.certainty import studies_the_link
from causal_genie.agent.state import RunState, Verdict

log = logging.getLogger(__name__)

# Exposures computed from the date: nothing about the people studied can change them.
CALENDAR_RULES = {"full_moon", "day_of_week", "weekend", "holiday", "season"}


def pick_verdict(state: RunState) -> Verdict:
    """The verdict for a finished run; the report only renders it."""
    if state.route is not None and state.route.kind == "not_causal":
        return "not_causal"
    if state.results:
        if unmeasured_confounders(state):
            return "estimated_strong_assumptions"
        return "estimated"
    if state.design is not None and not state.design.estimable:
        return "study_plan"
    accepted = any(c.accepted for c in state.datasets)
    if not accepted and any(c.access == "restricted" for c in state.datasets):
        return "restricted"
    if common_causes(state):
        return "common_cause"
    if state.certainty is None or state.certainty.level == "very_low":
        return "insufficient"  # nothing included, or only sparse or conflicting evidence
    return "published_only"


def cited_edges(state: RunState) -> list[tuple[str, str]]:
    """Edges that cite a `documented` claim (quote found and judged to support). An assumed
    edge is the model's guess, and the question edge is the hypothesis, so neither counts."""
    documented = {c.id for c in state.claims if c.status == "documented"}
    return [
        (e.source, e.target) for e in state.edges if documented & {*e.claim_ids} and not e.question
    ]


def cited_paths(state: RunState) -> list[list[str]]:
    """Every path of cited edges from the exposure to the outcome (through mediators): the
    sources describe how the exposure could act, which is not proof that it does."""
    if state.query is None:
        return []
    x, y = state.query.exposure.name, state.query.outcome.name
    graph = nx.DiGraph(cited_edges(state))
    if x not in graph or y not in graph:
        return []
    return sorted(nx.all_simple_paths(graph, x, y))


def outcome_factors(state: RunState) -> list[str]:
    """Other things checked sources say drive the outcome (product, store), so the answer may
    depend on them. Nodes on a cited path from the exposure and shared causes are left out:
    the report names those in their own entries."""
    if state.query is None:
        return []
    x, y = state.query.exposure.name, state.query.outcome.name
    named = {n for p in cited_paths(state) for n in p} | {*common_causes(state), x}
    return sorted({s for s, t in cited_edges(state) if t == y and s not in named})


def common_causes(state: RunState) -> list[str]:
    """Nodes that checked sources say drive both the exposure and the outcome, directly or
    through other nodes (hot weather -> beach crowds -> shark bites), when no included study
    (trial or observational, not news of one) tests the link itself.

    Paths use only cited edges (`cited_edges`) and never pass through the exposure. A cited
    path from the exposure to the outcome rules it out: the sources then say the exposure may
    matter, so the link is not explained away. Any direct study rules it out too, since real
    effects have confounders as well and then the studies speak for themselves."""
    if state.query is None:
        return []
    if any(studies_the_link(s) for s in state.studies):
        return []
    if paths := cited_paths(state):
        log.info("common causes: none, cited paths from the exposure: %s", paths)
        return []
    x, y = state.query.exposure.name, state.query.outcome.name
    # A path through x is not a backdoor.
    cited = nx.DiGraph([(s, t) for s, t in cited_edges(state) if s != x])
    if x not in cited or y not in cited:
        return []
    both = sorted(nx.ancestors(cited, x) & nx.ancestors(cited, y))
    log.info("common causes: %s", both)
    return both


def words(text: str) -> set[str]:
    """Lowercase words with a plural `s` dropped, so `holidays` matches `holiday`."""
    return {w.removesuffix("s") for w in re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))}


def unmeasured_confounders(state: RunState) -> list[str]:
    """Confounders the literature names that the analysis table lacks; only for an
    observational (backdoor) estimate, since randomization balances confounders.

    Two sources: DAG nodes with edges into both exposure and outcome that are not mapped to a
    column, and variables that included studies adjusted for (only from claims that passed
    the citation checks) matching no mapped node or column. A name matches when one's words
    contain the other's (e.g. `smoking status` ~ `smoking`); unmatched names are kept, since
    missing a confounder is the error this check guards against.

    An exposure read off the calendar (moon phase, weekday, holiday, season) has none:
    studies may adjust for parity or age, but nothing about the people studied changes a date.
    """
    observational = state.results and state.results[-1].tool == "backdoor_effect"
    if not observational or state.query is None or state.mapping is None:
        return []
    exposure, outcome = state.query.exposure.name, state.query.outcome.name
    mapped = [m for m in state.mapping.nodes if m.columns]
    if any(m.node == exposure and m.derive in CALENDAR_RULES for m in mapped):
        log.info("unmeasured confounders: none, %s is read off the calendar", exposure)
        return []
    names = [words(n) for m in mapped for n in (m.node, *m.columns)]

    def covered(name: str) -> bool:
        w = words(name)
        return any(w <= n or n <= w for n in names if n)

    into = {(e.source, e.target) for e in state.edges}
    in_dag = sorted(
        {s for s, t in into if t == exposure and (s, outcome) in into and not covered(s)}
    )
    checked = {
        c.study_id for c in state.claims if c.kind == "confounder" and c.status == "documented"
    }
    named = sorted(
        {
            item
            for s in state.studies
            if s.included and s.id in checked
            for item in s.adjusted_for
            if not covered(item)
        }
    )
    missing = in_dag + [n for n in named if n not in in_dag]
    log.info("unmeasured confounders: %s", missing)
    return missing
