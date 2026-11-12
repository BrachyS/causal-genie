"""Mapping step (PRD M6): which columns of the accepted table hold which DAG nodes.

The small model proposes the mapping from column names and summaries only (rows never go to a
model), starting from data discovery's coverage guess. The harness then checks it: columns
must exist, derive rules must come from the fixed menu with the inputs they need, and only
DAG nodes count. The user confirms; with nobody to ask, the mapping is kept but marked
unconfirmed so the report can say so.
"""

import logging
from collections.abc import Callable

from causal_genie.agent.state import Mapping, NodeMapping, RunState
from causal_genie.agent.steps.data import dag_nodes
from causal_genie.llm.client import LLMClient
from causal_genie.log import log_step
from causal_genie.tools.derive import DATE_RULES

log = logging.getLogger(__name__)


def unmapped(node: str) -> NodeMapping:
    return NodeMapping(node=node, columns=[], derive=None, cutoff=None, categorical=False)


def check_node(m: NodeMapping, columns: set[str]) -> NodeMapping:
    """The node as proposed if its columns exist and fit its rule; otherwise unmapped (logged)."""
    if unknown := [c for c in m.columns if c not in columns]:
        problem = f"unknown columns {unknown}"
    elif m.derive in DATE_RULES and len(m.columns) not in (0, 1, 3):
        problem = f"rule {m.derive} needs one date column or year, month, day"
    elif m.derive == "cutoff" and (m.cutoff is None or len(m.columns) != 1):
        problem = "rule cutoff needs one column and a cutoff value"
    elif m.derive is None and len(m.columns) > 1:
        problem = f"several columns {m.columns} but no rule to combine them"
    else:
        return m
    log.warning("mapping: %s unmapped: %s", m.node, problem)
    return unmapped(m.node)


def check_mapping(proposed: Mapping, nodes: list[str], columns: list[str]) -> Mapping:
    """One checked entry per DAG node, in DAG order; unknown nodes and panel columns dropped."""
    given = {m.node: m for m in proposed.nodes}
    if extra := sorted(set(given) - set(nodes)):
        log.warning("mapping: ignored nodes not in the DAG %s", extra)
    names = set(columns)
    panel = {}
    for role, column in (("unit", proposed.unit), ("time", proposed.time)):
        if column is not None and column not in names:
            log.warning("mapping: %s column %r not in the table", role, column)
            column = None
        panel[role] = column
    checked = [check_node(given[n], names) if n in given else unmapped(n) for n in nodes]
    return Mapping(nodes=checked, unit=panel["unit"], time=panel["time"])


def describe(m: NodeMapping) -> str:
    """One mapping line for the user, e.g. `sore_throat ← pain > 0`."""
    if not m.columns:
        return f"{m.node} ← (not in the table)"
    if m.derive == "cutoff":
        return f"{m.node} ← {m.columns[0]} > {m.cutoff:g}"
    if m.derive is not None:
        return f"{m.node} ← {m.derive}({', '.join(m.columns)})"
    return f"{m.node} ← {m.columns[0]}" + (" (categories)" if m.categorical else "")


def map_columns(
    state: RunState, llm: LLMClient, ask: Callable[[str], str] | None
) -> Mapping | None:
    """Propose, check and confirm the column ↔ node mapping; saved on `state.mapping`.

    Returns None when the user declines it.
    """
    assert state.query is not None and state.profile is not None
    nodes = dag_nodes(state)
    accepted = next(c for c in state.datasets if c.accepted)
    guess = "\n".join(f"- {n}: {accepted.coverage.get(n) or '(none)'}" for n in nodes)
    columns = "\n".join(
        f"- {c.name} ({c.dtype}; {c.missing} missing; {c.unique} distinct"
        + (f"; min {c.min:g}, max {c.max:g}, mean {c.mean:g})" if c.mean is not None else ")")
        for c in state.profile.columns
    )
    user = (
        f"Exposure: {state.query.exposure.name} ({state.query.exposure.definition})\n"
        f"Outcome: {state.query.outcome.name} ({state.query.outcome.definition})\n"
        f"Population: {state.query.population}\n"
        f"DAG nodes: {', '.join(nodes)}\n\n"
        f"Earlier coverage guess:\n{guess}\n\n"
        f"Table: {accepted.title}, {state.profile.rows} rows\nColumns:\n{columns}"
    )
    with log_step("map", nodes=nodes, table=accepted.url) as rec:
        proposed = llm.complete("small", "map", user, Mapping)
        mapping = check_mapping(proposed, nodes, [c.name for c in state.profile.columns])
        rec["output"] = mapping.model_dump()
    if ask is None:
        log.warning("mapping not confirmed: no user to ask")
        state.mapping, state.mapping_confirmed = mapping, False
        return mapping
    lines = "\n".join(f"  {describe(m)}" for m in mapping.nodes)
    panel = f"\n  panel: unit = {mapping.unit}, time = {mapping.time}" if mapping.unit else ""
    reply = ask(f"Column mapping for {accepted.title}:\n{lines}{panel}\nUse this mapping? (yes/no)")
    if not reply.strip().lower().startswith("y"):
        log.info("mapping declined by the user: %r", reply)
        state.mapping, state.mapping_confirmed = None, False
        return None
    state.mapping, state.mapping_confirmed = mapping, True
    return mapping
