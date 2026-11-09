"""DAG step (PRD M4): the large model proposes causal edges from the claim ledger.

The model may only cite claims the harness shows it, and it is only shown `documented` claims
(quote found in the source and judged to support the claim). Any other id it cites is dropped
and logged; an edge left with no citation is "assumed". The question's own exposure → outcome
edge is always in the graph, so identification has both nodes to work with. A node whose name
is a near copy of the exposure's or outcome's (e.g. `licorice_gargling`) is logged: it usually
means the model renamed a question node, which would leave a stray node in the graph.
"""

import logging

from rapidfuzz import fuzz

from causal_genie.agent.state import DagEdge, DagProposal, RunState
from causal_genie.llm.client import LLMClient

log = logging.getLogger(__name__)

# rapidfuzz ratio; at or above this, a node name different from a question node is a near copy
NEAR_COPY = 85


def build_dag(state: RunState, llm: LLMClient) -> list[DagEdge]:
    """Edges for the question's DAG, each citing documented ledger claims or marked assumed."""
    assert state.query is not None, "frame the question before building the DAG"
    exposure, outcome = state.query.exposure, state.query.outcome
    citable = [c for c in state.claims if c.status == "documented"]
    claims = "\n\n".join(f"Claim {c.id} ({c.kind}): {c.text}\nQuote: {c.quote}" for c in citable)
    user = (
        f"Exposure: {exposure.name} ({exposure.definition})\n"
        f"Outcome: {outcome.name} ({outcome.definition})\n"
        f"Population: {state.query.population}\n\n"
        f"Claims:\n\n{claims or '(none)'}"
    )
    proposed = llm.complete("large", "dag", user, DagProposal)

    known = {c.id for c in citable}
    edges: list[DagEdge] = []
    for edge in proposed.edges:
        dropped = [i for i in edge.claim_ids if i not in known]
        if dropped:
            log.warning(
                "edge %s -> %s cites unknown or undocumented claims %s; dropped",
                edge.source,
                edge.target,
                dropped,
            )
        cited = [i for i in edge.claim_ids if i in known]
        edges.append(edge.model_copy(update={"claim_ids": cited}))

    nodes = {n for e in edges for n in (e.source, e.target)}
    for node in sorted(nodes - {exposure.name, outcome.name}):
        for name in (exposure.name, outcome.name):
            if fuzz.ratio(node, name) >= NEAR_COPY:
                log.warning("node %s looks like a renamed %s", node, name)

    if not any((e.source, e.target) == (exposure.name, outcome.name) for e in edges):
        log.warning("model left out %s -> %s; added as assumed", exposure.name, outcome.name)
        edges.append(DagEdge(source=exposure.name, target=outcome.name, claim_ids=[]))
    log.info("dag: %d edges, %d assumed", len(edges), sum(e.assumed for e in edges))
    return edges
