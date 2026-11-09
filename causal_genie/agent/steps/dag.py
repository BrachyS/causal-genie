"""DAG step (PRD M4): the large model proposes causal edges from the claim ledger.

The model may only cite claims the harness shows it, and it is only shown `documented` claims
(quote found in the source and judged to support the claim). Any other id it cites is dropped
and logged; an edge left with no citation is "assumed". The question's own exposure → outcome
edge is always in the graph, so identification has both nodes to work with. A node whose name
is a near copy of the exposure's or outcome's (e.g. `licorice_gargling`) is logged: it usually
means the model renamed a question node, which would leave a stray node in the graph.

The harness then makes the graph a DAG: each cycle loses its weakest-evidence edge. An edge
with no citation sends research back for one more round if rounds are left; otherwise it stays
assumed.
"""

import logging

import networkx as nx
from rapidfuzz import fuzz

from causal_genie.agent.state import Claim, DagEdge, DagProposal, RunState
from causal_genie.agent.steps.research import MAX_ROUNDS, research_gaps
from causal_genie.llm.client import LLMClient
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.search import Searcher

log = logging.getLogger(__name__)

# rapidfuzz ratio; at or above this, a node name different from a question node is a near copy
NEAR_COPY = 85


def run_dag(
    state: RunState,
    llm: LLMClient,
    searcher: Searcher,
    papers: EuropePMC,
    max_rounds: int = MAX_ROUNDS,
) -> None:
    """Set `state.edges`, sending uncited edges back to research once if rounds are left."""
    state.edges = build_dag(state, llm)
    uncited = [e for e in state.edges if e.assumed]
    if not uncited:
        return
    if len(state.rounds) >= max_rounds:
        log.info("no research rounds left; %d edges stay assumed", len(uncited))
        return
    claims_before = citable_claims(state)
    gaps = [f"no source yet for DAG edge {e.source} -> {e.target}" for e in uncited]
    research_gaps(state, llm, searcher, papers, gaps, max_rounds)
    if citable_claims(state) != claims_before:
        state.edges = build_dag(state, llm)


def citable_claims(state: RunState) -> list[Claim]:
    """The ledger claims an edge may cite: quote found in the source and judged to support."""
    return [c for c in state.claims if c.status == "documented"]


def break_cycles(edges: list[DagEdge], question: tuple[str, str]) -> list[DagEdge]:
    """The edges with every cycle broken, each by dropping (and logging) its weakest edge.

    Weakest = fewest cited claims, so assumed edges go first; ties go by name, so a run does
    not depend on edge order. The `question` edge (exposure, outcome) goes last of all.
    """
    kept = list(edges)
    while not nx.is_directed_acyclic_graph(
        graph := nx.DiGraph([(e.source, e.target) for e in kept])
    ):
        cycle = set(nx.find_cycle(graph))
        weakest = min(
            (e for e in kept if (e.source, e.target) in cycle),
            key=lambda e: ((e.source, e.target) == question, len(e.claim_ids), e.source, e.target),
        )
        log.warning(
            "cycle %s: dropped %s -> %s (%d claims)",
            sorted(cycle),
            weakest.source,
            weakest.target,
            len(weakest.claim_ids),
        )
        kept.remove(weakest)
    return kept


def build_dag(state: RunState, llm: LLMClient) -> list[DagEdge]:
    """Edges for the question's DAG, each citing documented ledger claims or marked assumed."""
    assert state.query is not None, "frame the question before building the DAG"
    exposure, outcome = state.query.exposure, state.query.outcome
    citable = citable_claims(state)
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
    edges = break_cycles(edges, (exposure.name, outcome.name))
    log.info("dag: %d edges, %d assumed", len(edges), sum(e.assumed for e in edges))
    return edges
