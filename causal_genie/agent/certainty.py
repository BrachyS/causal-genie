"""Certainty label and evidence-flow counts: rules in code, never a model's judgment.

Certainty (PRD §5.4, inspired by GRADE) starts from the best design among included studies
and steps down for disagreement and indirectness. A review counts at the level of what it
pools: a review of observational studies is observational evidence. Conflicts lower
certainty and are explained in the report; studies are never vote-counted.
"""

from collections import Counter
from typing import get_args

from causal_genie.agent.state import Certainty, CertaintyLevel, EvidenceFlow, Study

LEVELS: tuple[CertaintyLevel, ...] = get_args(CertaintyLevel)  # lowest first
REVIEW_DESIGNS = {"meta_analysis", "systematic_review"}
OBSERVATIONAL_DESIGNS = {"natural_experiment", "cohort", "case_control", "cross_sectional"}
DIRECTIONS = {"+", "-", "null"}  # mixed/unclear results can't disagree with anything


def design_tier(study: Study) -> int:
    """Strength of a study's design: 2 randomized trial or a review pooling only trials,
    1 observational or any other review, 0 anything else (case reports, other)."""
    if study.design == "RCT" or (study.design in REVIEW_DESIGNS and study.pooled == "trials"):
        return 2
    if study.design in REVIEW_DESIGNS or study.design in OBSERVATIONAL_DESIGNS:
        return 1
    return 0


def certainty(studies: list[Study]) -> Certainty | None:
    """Certainty of the included evidence, with a reason for each step; None if none included."""
    included = [s for s in studies if s.included]
    if not included:
        return None
    papers = [s for s in included if not s.secondary_only]
    trials = [s for s in papers if design_tier(s) == 2]
    observational = [s for s in papers if design_tier(s) == 1]
    if trials:
        level, best, reasons = 3, trials, ["starts high: a trial or a review of trials matches"]
    elif observational:
        level, best, reasons = 1, observational, ["starts low: best evidence is observational"]
    else:
        level, best = 0, included
        reasons = ["very low: only case reports, other designs or secondary sources"]

    directions = {s.direction for s in best} & DIRECTIONS
    if len(directions) > 1 and level > 0:
        level -= 1
        shown = ", ".join(sorted(directions))
        reasons.append(f"down one: best studies point in different directions ({shown})")
    if all(s.peco_match == "indirect" for s in best) and level > 0:
        level -= 1
        reasons.append("down one: best evidence is about a different population or exposure")
    return Certainty(level=LEVELS[level], reasons=reasons)


def evidence_flow(studies: list[Study]) -> EvidenceFlow:
    """Sources found → studies after dedupe → included, with exclusion reasons."""
    excluded = Counter(s.exclusion_reason for s in studies if not s.included)
    return EvidenceFlow(
        found=sum(1 + len(s.also_at) for s in studies),
        after_dedupe=len(studies),
        included=sum(s.included for s in studies),
        excluded={str(reason): n for reason, n in excluded.items()},
    )
