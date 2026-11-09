"""Identification tools: decide what to adjust for from the DAG and the data's design.

The DAG step draws the real world, where anything might influence who gets the exposure. Data
from a randomized trial is different: assignment alone sets the exposure, so arrows into it do
not hold there and must not be used to pick an adjustment set.
"""

import logging

from causal_genie.agent.state import DagEdge

log = logging.getLogger(__name__)


def randomized_graph(edges: list[DagEdge], exposure: str) -> list[DagEdge]:
    """The DAG as it holds in randomized data: every edge into the exposure removed and logged."""
    for e in edges:
        if e.target == exposure:
            log.info("randomized data: dropped %s -> %s", e.source, e.target)
    return [e for e in edges if e.target != exposure]
