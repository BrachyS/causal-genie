"""Pick the run's verdict (PRD §5.2) from what the run found. Rules only, never a model.

The verdict reads the run state after every step has run: an estimate beats a study plan,
a study plan beats "no data", and "no data" splits into restricted, published-only or
insufficient by what research and data discovery found.
"""

import logging
import re

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
    if state.certainty is None or state.certainty.level == "very_low":
        return "insufficient"  # nothing included, or only sparse or conflicting evidence
    return "published_only"


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
