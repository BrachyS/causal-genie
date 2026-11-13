"""Identification tools: decide what the data can estimate and what to adjust for.

The DAG step draws the real world, where anything might influence who gets the exposure. Data
from a randomized trial is different: assignment alone sets the exposure, so arrows into it do
not hold there and must not be used to pick an adjustment set.

`detect_design` labels the data's design from the table and the DAG, never from a model. The
MVP estimates randomized and backdoor-adjusted data only; DiD, IV and RDD setups are named
with their key assumption and the check to run, but not estimated.
"""

import logging

import networkx as nx
import pandas as pd
from dowhy.causal_identifier.auto_identifier import (
    EstimandType,
    get_instruments,
    identify_effect_auto,
)
from scipy import stats

from causal_genie.agent.state import Adjustment, DagEdge, DesignCheck

log = logging.getLogger(__name__)


def randomized_graph(edges: list[DagEdge], exposure: str) -> list[DagEdge]:
    """The DAG as it holds in randomized data: every edge into the exposure removed and logged."""
    for e in edges:
        if e.target == exposure:
            log.info("randomized data: dropped %s -> %s", e.source, e.target)
    return [e for e in edges if e.target != exposure]


def backdoor_set(
    graph: nx.DiGraph, exposure: str, outcome: str, observed: list[str]
) -> list[str] | None:
    """DoWhy's default backdoor set using only `observed` nodes; None if none is valid."""
    estimand = identify_effect_auto(
        graph, [exposure], [outcome], observed, EstimandType.NONPARAMETRIC_ATE
    )
    if (estimand.estimands or {}).get("backdoor") is None:
        return None
    return sorted(estimand.get_backdoor_variables())


def valid_adjustment(graph: nx.DiGraph, exposure: str, outcome: str, adjust_for: list[str]) -> bool:
    """The backdoor criterion: nothing in the set is caused by the exposure, and the set blocks
    every path from exposure to outcome that starts with an arrow into the exposure."""
    if set(adjust_for) & nx.descendants(graph, exposure):
        return False
    backdoor = graph.copy()
    backdoor.remove_edges_from(list(graph.out_edges(exposure)))
    return nx.is_d_separator(backdoor, {exposure}, {outcome}, set(adjust_for))


def precision_set(
    edges: list[DagEdge], exposure: str, outcome: str, adjust_for: list[str], observed: list[str]
) -> list[str]:
    """Observed direct causes of the outcome that the exposure does not affect, to adjust for
    on top of `adjust_for`: they explain outcome variation, so the CI narrows without bias.

    Each is kept only if the enlarged set still passes the backdoor criterion (one that would
    open a path is logged and left out). Returns the added names, in name order.
    """
    graph = nx.DiGraph([(e.source, e.target) for e in edges])
    graph.add_nodes_from([exposure, outcome])
    after = nx.descendants(graph, exposure)
    chosen = list(adjust_for)
    for node in sorted(graph.predecessors(outcome)):
        if node == exposure or node in after or node in chosen or node not in observed:
            continue
        if valid_adjustment(graph, exposure, outcome, [*chosen, node]):
            chosen.append(node)
        else:
            log.info("precision: %s left out; adjusting for it would open a backdoor path", node)
    added = chosen[len(adjust_for) :]
    log.info("precision: adding %s to %s", added, adjust_for)
    return added


def identify(
    edges: list[DagEdge], exposure: str, outcome: str, observed: list[str], randomized: bool
) -> Adjustment:
    """The backdoor adjustment set among the nodes the table has (`observed`).

    With `randomized`, edges into the exposure are cut first, so nothing needs adjusting for.
    When no valid set exists, `missing` names the unobserved nodes the set would need, so the
    report can say which confounder the data lacks.
    """
    if missing_core := [n for n in (exposure, outcome) if n not in observed]:
        raise ValueError(f"exposure and outcome must be observed; missing {missing_core}")
    used = randomized_graph(edges, exposure) if randomized else edges
    graph = nx.DiGraph([(e.source, e.target) for e in used])
    graph.add_nodes_from([exposure, outcome])
    seen = [n for n in graph.nodes if n in observed]
    adjust_for = backdoor_set(graph, exposure, outcome, seen)
    missing = []
    if adjust_for is None:
        needed = backdoor_set(graph, exposure, outcome, list(graph.nodes)) or []
        missing = sorted(set(needed) - set(observed))
    log.info("identify: adjust_for=%s missing=%s randomized=%s", adjust_for, missing, randomized)
    return Adjustment(adjust_for=adjust_for, missing=missing, randomized=randomized)


ROSTER_SHORTFALL = 0.05  # rows may fall short of the randomized roster by this share (dropouts)
SRM_P = 0.001  # arm sizes this unlikely under the planned allocation mean rows were selected
MIN_SCORE_VALUES = 20  # a running variable for RDD needs many distinct values


def selection_reasons(
    table: pd.DataFrame, exposure: str, roster: int | None, allocation: float
) -> list[str]:
    """Signs that rows were kept or dropped after assignment, so arms are no longer comparable:
    far fewer rows than were randomized, or arm sizes off the planned allocation (a sample
    ratio mismatch). One missing patient in a trial is not selection."""
    reasons = []
    arm = table[exposure].dropna()
    if roster is not None and len(table) < (1 - ROSTER_SHORTFALL) * roster:
        reasons.append(f"{len(table)} rows but {roster} were randomized")
    treated = int(arm.sum())
    p = stats.binomtest(treated, len(arm), allocation).pvalue
    if p < SRM_P:
        reasons.append(
            f"arm sizes {treated} treated vs {len(arm) - treated} control are off the planned "
            f"{allocation:.0%} allocation (p = {p:.1g})"
        )
    return reasons


def running_variable(
    table: pd.DataFrame, exposure: str, skip: set[str | None]
) -> tuple[str, float] | None:
    """A numeric column that alone sets the exposure at a cutoff: (column, cutoff) or None."""
    x = table[exposure]
    for column in table.columns:
        values = table[column]
        if column in skip or not pd.api.types.is_numeric_dtype(values):
            continue
        if values.nunique() < MIN_SCORE_VALUES:
            continue
        treated, control = values[x == 1], values[x == 0]
        if len(treated) and len(control) and treated.min() > control.max():
            return column, float(treated.min())
        if len(treated) and len(control) and treated.max() < control.min():
            return column, float(treated.max())
    return None


def staggered_panel(table: pd.DataFrame, exposure: str, unit: str, time: str) -> bool:
    """Units observed before and after their exposure starts, with others not yet (or never)
    exposed to compare with: the DiD setup."""
    ordered = table.sort_values(time)
    starts = ordered[ordered[exposure] == 1].groupby(unit)[time].min()
    ever = set(starts.index)
    switchers = [u for u in ever if (ordered.loc[ordered[unit] == u, exposure] == 0).any()]
    never = set(table[unit]) - ever
    return bool(switchers) and (bool(never) or starts.nunique() > 1)


def detect_design(
    table: pd.DataFrame,
    exposure: str,
    outcome: str,
    edges: list[DagEdge],
    randomized: bool,
    roster: int | None = None,
    allocation: float = 0.5,
    unit: str | None = None,
    time: str | None = None,
) -> DesignCheck:
    """Label the data's design: randomized only if assignment is documented (`randomized`) and
    is the exposure column; otherwise RDD, DiD, IV or plain observational data.

    `table` has one column per mapped node (plus a panel's `unit` and `time`); `roster` is the
    number randomized when the source states it.
    """
    if randomized:
        if reasons := selection_reasons(table, exposure, roster, allocation):
            design = DesignCheck(
                design="selection_after_assignment",
                estimable=False,
                key_assumption="arms stay comparable only if every randomized unit is kept",
                estimand="none: comparing the kept rows is biased",
                check="ask for the full randomized roster, including units that left early",
                reasons=reasons,
            )
        else:
            design = DesignCheck(
                design="randomized",
                estimable=True,
                key_assumption="assignment was random and the table holds the randomized roster",
                estimand="ATE",
                check="arm sizes match the planned allocation",
                reasons=["assignment is documented and recorded in the exposure column"],
            )
        log.info("detect_design: %s", design)
        return design

    skip = {exposure, outcome, unit, time}
    if found := running_variable(table, exposure, skip):
        column, cutoff = found
        design = DesignCheck(
            design="RDD",
            estimable=False,
            key_assumption=f"units just either side of {column} = {cutoff:g} are alike",
            estimand="effect at the cutoff",
            check=f"no bunching of {column} at the cutoff; covariates smooth across it",
            reasons=[f"{exposure} is set by {column} crossing {cutoff:g}"],
        )
    elif unit and time and staggered_panel(table, exposure, unit, time):
        design = DesignCheck(
            design="DiD",
            estimable=False,
            key_assumption="parallel trends: without exposure, exposed and comparison units "
            "would have changed alike",
            estimand="ATT",
            check="pre-exposure trends of exposed and comparison units match (event study)",
            reasons=[f"{unit}s observed before and after {exposure} starts, with comparisons"],
        )
    else:
        design = observational(table, exposure, outcome, edges)
    log.info("detect_design: %s", design)
    return design


def observational(
    table: pd.DataFrame, exposure: str, outcome: str, edges: list[DagEdge]
) -> DesignCheck:
    """Backdoor adjustment if the table has a valid set; else IV if a cited instrument is in
    the table; else not estimable, naming the confounders the data lacks."""
    observed = list(table.columns)
    adj = identify(edges, exposure, outcome, observed, randomized=False)
    if adj.adjust_for is not None:
        named = ", ".join(adj.adjust_for) or "nothing (no backdoor path in the DAG)"
        return DesignCheck(
            design="observational",
            estimable=True,
            key_assumption=f"no unmeasured confounding after adjusting for {named}",
            estimand="ATE",
            check="compare adjusted and naive estimates; name confounders the DAG may miss",
            reasons=["no randomization documented; the DAG gives a backdoor set the table has"],
        )
    graph = nx.DiGraph([(e.source, e.target) for e in edges])
    cited = {e.source for e in edges if e.target == exposure and not e.assumed}
    instruments = sorted(set(get_instruments(graph, [exposure], [outcome])) & cited & set(observed))
    if instruments:
        return DesignCheck(
            design="IV",
            estimable=False,
            key_assumption=f"{', '.join(instruments)} affects {outcome} only through {exposure} "
            "and shares no cause with it",
            estimand="LATE",
            check=f"first-stage strength of {', '.join(instruments)} (F > 10) and the cited "
            "reason for the exclusion restriction",
            reasons=[
                f"confounders {', '.join(adj.missing)} are not in the table",
                f"{', '.join(instruments)}: cited cause of {exposure}, no other path to "
                f"{outcome} in the DAG",
            ],
        )
    return DesignCheck(
        design="observational",
        estimable=False,
        key_assumption="no unmeasured confounding",
        estimand="none: confounding cannot be removed with this table",
        check=f"find data that also measures {', '.join(adj.missing)}",
        reasons=[f"confounders {', '.join(adj.missing)} are not in the table"],
    )
