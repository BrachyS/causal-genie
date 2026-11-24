"""DAG step (PRD M4): the large model proposes causal edges from the claim ledger.

The model may only cite claims the harness shows it, and it is only shown `documented` claims
(quote found in the source and judged to support the claim). Any other id it cites is dropped
and logged; an edge left with no citation is "assumed". A `null_result` claim ("no effect
found") is shown to the model but never supports an edge: citing it is dropped and logged the
same way. The question's own exposure → outcome
edge is always in the graph, so identification has both nodes to work with. It is the
hypothesis being tested, not a fact to cite: it is marked `question`, never cites claims (any
the model gives are dropped and logged) and never counts as assumed. Its evidence, supporting
and null, is the report's Evidence section. A node whose name
is a near copy of the exposure's or outcome's (e.g. `licorice_gargling`) is logged: it usually
means the model renamed a question node, which would leave a stray node in the graph.

The model also names exogenous nodes: ones nothing in the graph can cause (moon phase, weather,
calendar date, a coin flip). The harness drops and logs every edge into them, so a wrong arrow
such as `season -> full_moon` cannot turn a calendar factor into a fake confounder. Edges out of
them stay; a name not in the graph is logged and ignored. The nodes and the dropped edges are
kept in the run state, so the report can show them as the model's assumption.

A claim backing its own quote does not make it evidence for an arrow: "discounts drive impulse
buying" supports `discounts -> impulse_buying`, not the reverse. So the large model (reasoning
off) judges every remaining (edge, cited claim) pair in one call, and the harness keeps a claim
on an edge only if it `supports` that direction. Reversed, direction-free or skipped pairs are
dropped and logged, so the edge may become assumed and no longer counts as evidence (e.g. for a
common cause).

The harness then makes the graph a DAG: each cycle loses its weakest-evidence edge. An edge
with no citation out of a cause of the exposure (a possible backdoor path or instrument) sends
research back for one more round if rounds are left; every other uncited edge stays assumed.
"""

import logging

import networkx as nx
from rapidfuzz import fuzz

from causal_genie.agent.state import Claim, DagEdge, DagProposal, EdgeCheck, RunState
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
    """Set `state.edges`, sending uncited edges back to research once if rounds are left.

    Only edges out of a cause of the exposure are sent: they may lie on a backdoor path, so
    they decide what to adjust for (and a lone edge into the exposure may be an instrument).
    Other edges, e.g. weekday -> births, season -> weather or exposure -> mediator, cannot change
    the adjustment set, so a source for them is not worth a round.
    """
    assert state.query is not None
    state.edges = build_dag(state, llm)
    causes = exposure_causes(state.edges, state.query.exposure.name)
    uncited = [e for e in state.edges if e.assumed and e.source in causes]
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


def exposure_causes(edges: list[DagEdge], exposure: str) -> set[str]:
    """Every node with a directed path into the exposure (its ancestors in the DAG)."""
    graph = nx.DiGraph([(e.source, e.target) for e in edges])
    return nx.ancestors(graph, exposure) if exposure in graph else set()


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


def drop_into_exogenous(
    edges: list[DagEdge], exogenous: list[str], question: tuple[str, str]
) -> tuple[list[str], list[DagEdge], list[DagEdge]]:
    """Split off (and log) every edge into a node the model says nothing can cause.

    Returns the exogenous names found in the graph, the edges kept and the edges dropped. The
    question edge is never dropped: it is the hypothesis being tested.
    """
    nodes = {n for e in edges for n in (e.source, e.target)}
    for name in exogenous:
        if name not in nodes:
            log.warning("exogenous node %s is not in the graph; ignored", name)
    found = [name for name in dict.fromkeys(exogenous) if name in nodes]
    kept, dropped = [], []
    for e in edges:
        if e.target not in found:
            kept.append(e)
        elif (e.source, e.target) == question:
            log.warning("%s called exogenous; kept the question edge %s -> %s", e.target, *question)
            kept.append(e)
        else:
            log.info("exogenous %s: dropped %s -> %s", e.target, e.source, e.target)
            dropped.append(e)
    return found, kept, dropped


def check_edges(edges: list[DagEdge], claims: dict[str, Claim], llm: LLMClient) -> list[DagEdge]:
    """The edges keeping only the claims whose quote says source -> target in that direction.

    The model judges each (edge, claim) pair; the harness drops (and logs) every pair not
    judged `supports`, including pairs the model skipped. Pairs are numbered in sorted order,
    so the call's input, and so its replay, does not depend on the order of the edges.
    """
    pairs = sorted((e.source, e.target, i) for e in edges for i in e.claim_ids)
    if not pairs:
        return edges
    ids = {f"p{n}": pair for n, pair in enumerate(pairs, 1)}
    user = "\n\n".join(
        f"Pair {pid}: {s} -> {t}\nClaim {i}: {claims[i].text}\nQuote: {claims[i].quote}"
        for pid, (s, t, i) in ids.items()
    )
    check = llm.complete("large", "check_edge", user, EdgeCheck)
    judged = {ids[j.pair_id]: j for j in check.judgments if j.pair_id in ids}
    if unknown := [j.pair_id for j in check.judgments if j.pair_id not in ids]:
        log.warning("edge check judged unknown pairs %s; ignored", unknown)
    if skipped := [pid for pid, pair in ids.items() if pair not in judged]:
        log.warning("edge check skipped pairs %s; their claims are dropped", skipped)
    kept = []
    for e in edges:
        verdicts = {i: judged.get((e.source, e.target, i)) for i in e.claim_ids}
        dropped = {
            i: (j.support, j.reason) if j else ("skipped", "")
            for i, j in verdicts.items()
            if j is None or j.support != "supports"
        }
        if dropped:
            log.warning(
                "edge %s -> %s: claims do not support it in this direction; dropped %s",
                e.source,
                e.target,
                dropped,
            )
        kept.append(
            e.model_copy(update={"claim_ids": [i for i in e.claim_ids if i not in dropped]})
        )
    return kept


def build_dag(state: RunState, llm: LLMClient) -> list[DagEdge]:
    """Edges for the question's DAG, each citing documented ledger claims or marked assumed.

    Also records in `state` the exogenous nodes and the edges dropped into them.
    """
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
    nulls = {c.id for c in citable if c.kind == "null_result"}
    question = (exposure.name, outcome.name)
    edges: list[DagEdge] = []
    for edge in proposed.edges:
        is_question = (edge.source, edge.target) == question
        dropped = [i for i in edge.claim_ids if i not in known]
        if dropped:
            log.warning(
                "edge %s -> %s cites unknown or undocumented claims %s; dropped",
                edge.source,
                edge.target,
                dropped,
            )
        cited = [i for i in edge.claim_ids if i in known]
        if is_question and cited:
            log.info(
                "question edge %s -> %s is the hypothesis, not a cited fact; dropped claims %s "
                "(they stay in the Evidence section)",
                edge.source,
                edge.target,
                cited,
            )
            cited = []
        if null_cited := [i for i in cited if i in nulls]:
            log.warning(
                "edge %s -> %s cites null result claims %s; a null result supports no edge; "
                "dropped",
                edge.source,
                edge.target,
                null_cited,
            )
            cited = [i for i in cited if i not in nulls]
        edges.append(edge.model_copy(update={"claim_ids": cited, "question": is_question}))

    nodes = {n for e in edges for n in (e.source, e.target)}
    for node in sorted(nodes - {exposure.name, outcome.name}):
        for name in (exposure.name, outcome.name):
            if fuzz.ratio(node, name) >= NEAR_COPY:
                log.warning("node %s looks like a renamed %s", node, name)

    if not any(e.question for e in edges):
        log.warning("model left out %s -> %s; added as the question edge", *question)
        edges.append(DagEdge(source=question[0], target=question[1], claim_ids=[], question=True))
    state.exogenous, edges, state.exogenous_dropped = drop_into_exogenous(
        edges, proposed.exogenous, question
    )
    edges = check_edges(edges, {c.id: c for c in citable}, llm)
    edges = break_cycles(edges, question)
    log.info("dag: %d edges, %d assumed", len(edges), sum(e.assumed for e in edges))
    return edges
