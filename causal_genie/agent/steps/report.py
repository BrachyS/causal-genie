"""The run's Markdown report (PRD §5.8): harness-rendered sections plus model-written prose.

Sections with facts (verdict, PECO, DAG, confounders, evidence, data, estimate) are rendered
here from run state, so their numbers are copied from `Result`s. The large model writes only
the summary of findings, the comparison with published results and the limitations, from a list of
ledger entries: documented claims, tool results and run facts. The harness lints each
sentence (it cites an entry; each number in it appears in a cited entry or in the question),
asks once for a rewrite naming the problems, then drops the sentences that still fail.
"""

import logging
import re

from causal_genie.agent.certainty import design_tier, evidence_flow
from causal_genie.agent.state import ReportDraft, RunState, Sentence, Verdict
from causal_genie.agent.verdict import common_causes, unmeasured_confounders
from causal_genie.llm.client import LLMClient

log = logging.getLogger(__name__)

SUMMARY_WORDS = 120
FINDINGS = "What the research and analysis found"  # the summary's heading
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

VERDICT_LABELS: dict[Verdict, str] = {
    "estimated": "✅ Estimated from data",
    "estimated_strong_assumptions": "⚠️ Estimated, strong assumptions",
    "published_only": "📚 Published evidence only",
    "restricted": "🔒 Data restricted",
    "study_plan": "🧪 Not estimated — study plan",
    "common_cause": "🔀 Explained by a common cause",
    "insufficient": "❓ Insufficient evidence",
    "not_causal": "🚫 Not causal",
}

DISCLAIMER = (
    "Causal Genie writes research summaries, not medical advice. Check with a qualified "
    "professional before acting on them."
)


def write_report(state: RunState) -> str:
    """Render the run as Markdown. The controller sets the verdict first, unless an API
    failure stopped the run: then the report is partial and names the failed step."""
    lines = [f"# {state.question}", ""]
    if state.failure is not None:
        lines += [
            f"**Partial report:** the `{state.step}` step failed ({state.failure}), so no "
            "verdict was reached. What follows is what ran before it. This is not a finding "
            "that no evidence exists.",
            "",
        ]
    else:
        assert state.verdict is not None, "the controller sets the verdict before the report"
        lines += [f"**Verdict:** {VERDICT_LABELS[state.verdict]}", ""]
    if state.verdict == "not_causal" and state.route is not None:
        lines += [state.route.reason, ""]
        if state.route.causal_rewrite:
            lines += [f"A causal question you might mean: *{state.route.causal_rewrite}*", ""]
    prose = state.prose or ReportDraft(summary=[], comparison=[], limitations=[])
    lines += paragraph(FINDINGS, prose.summary)
    lines += question_section(state) + dag_section(state) + confounders_section(state)
    lines += evidence_section(state) + data_section(state)
    if state.failure is None or state.results or state.design is not None:
        lines += estimate_section(state)
    lines += paragraph("Comparison with published", prose.comparison)
    if prose.limitations:
        cited = [f"- {s.text}{refs(s)}" for s in prose.limitations]
        lines += ["## Limitations", "", *cited, ""]
    if state.genie:
        lines += ["## 🧞 Genie's aside", "", state.genie, ""]
    lines += sources_section(state)
    lines += ["---", "", DISCLAIMER, "", f"Run ID: `{state.run_id}`", ""]
    return "\n".join(lines)


def refs(s: Sentence) -> str:
    """The sentence's claim and result citations, e.g. ` [c1, res-1a2b]`; run facts are
    shown in their own sections, so they are not listed."""
    shown = [c for c in s.cites if not c.startswith("F-")]
    return f" [{', '.join(shown)}]" if shown else ""


def paragraph(title: str, sentences: list[Sentence]) -> list[str]:
    """A prose section, each sentence followed by its citations; nothing if empty."""
    if not sentences:
        return []
    return [f"## {title}", "", " ".join(f"{s.text}{refs(s)}" for s in sentences), ""]


def dag_section(state: RunState) -> list[str]:
    """Each DAG edge with what backs it (cited claims, the question itself, or nothing), then
    the nodes treated as exogenous and the arrows into them that were dropped."""
    if not state.edges:
        return []
    lines = ["## Causal graph (DAG)", ""]
    for e in state.edges:
        if e.question:
            backing = "the question being tested"
        elif e.assumed:
            backing = "assumed (no source)"
        else:
            backing = f"cites {', '.join(e.claim_ids)}"
        lines.append(f"- {e.source} → {e.target}: {backing}")
    if state.exogenous:
        lines += [
            "",
            "Treated as exogenous (the model's judgment that nothing in the graph can cause it; "
            f"no source): {', '.join(state.exogenous)}.",
        ]
        if state.exogenous_dropped:
            dropped = ", ".join(f"{e.source} → {e.target}" for e in state.exogenous_dropped)
            lines[-1] += f" Arrows into them were dropped: {dropped}."
    return [*lines, ""]


def confounders_section(state: RunState) -> list[str]:
    """DAG confounders (edges into both exposure and outcome), how the analysis handled them,
    and confounders the literature names that the data lacks."""
    if state.query is None or not state.edges:
        return []
    exposure, outcome = state.query.exposure.name, state.query.outcome.name
    pairs = {(e.source, e.target) for e in state.edges}
    found = sorted({s for s, t in pairs if t == exposure and (s, outcome) in pairs})
    lines = ["## Key confounders", ""]
    lines += [f"- {c}" for c in found] or ["No confounder in the DAG."]
    a = state.adjustment
    if a is not None and a.randomized:
        lines += [
            "",
            "Data are randomized: assignment balances confounders, so none is adjusted for.",
        ]
    elif a is not None and a.adjust_for is not None:
        lines += ["", f"Adjusted for: {', '.join(a.adjust_for) or 'nothing'}."]
    if state.verdict == "common_cause":
        shared = ", ".join(common_causes(state))
        lines += ["", f"Causes of both, from checked sources (no study tests the link): {shared}."]
    if lacking := unmeasured_confounders(state):
        lines += ["", f"Named in the literature but not in the data: {', '.join(lacking)}."]
    return [*lines, ""]


def sources_section(state: RunState) -> list[str]:
    """Every claim and result the prose cites: the verbatim quote and its URL, or the tool."""
    cited = (
        [
            c
            for s in (state.prose.summary + state.prose.comparison + state.prose.limitations)
            for c in s.cites
        ]
        if state.prose
        else []
    )
    claims = {c.id: c for c in state.claims}
    studies = {s.id: s for s in state.studies}
    results = {r.id: r for r in state.results}
    lines = []
    for ref in dict.fromkeys(cited):
        if ref in results:
            lines.append(f"- [{ref}] tool `{results[ref].tool}` {results[ref].tool_version}")
        elif ref in claims and (study := studies.get(claims[ref].study_id or "")):
            lines.append(f'- [{ref}] {byline(study)}: "{claims[ref].quote}" — {study.url}')
    return ["## Sources", "", *lines, ""] if lines else []


def question_section(state: RunState) -> list[str]:
    """The framed estimand (PECO), with any ambiguities the user never resolved."""
    q = state.query
    if q is None:
        return []
    lines = [
        "## Question (PECO)",
        "",
        f"- **Population:** {q.population}",
        f"- **Exposure:** {q.exposure.definition}",
        f"- **Comparator:** {q.comparator}",
        f"- **Outcome:** {q.outcome.definition}",
        f"- **Time window:** {q.time_window or 'not specified'}",
        f"- **Estimand:** {q.target}, {estimand_scale(state)}, "
        f"deviations handled {q.deviation_strategy.replace('_', ' ')}",
    ]
    if q.ambiguities:
        lines.append(f"- **Unresolved ambiguities:** {'; '.join(q.ambiguities)}")
    return [*lines, ""]


def estimand_scale(state: RunState) -> str:
    """The effect scale the tool computed (the data decides it), noting the framed one when
    they differ, e.g. a count outcome framed as a risk difference."""
    assert state.query is not None
    framed = state.query.effect_scale.replace("_", " ")
    if not state.results:
        return framed
    computed = state.results[0].outputs["scale"].replace("_", " ")
    if computed == framed:
        return computed
    reason = "; the outcome in the data is not 0/1" if computed == "mean difference" else ""
    return f"{computed} (framed as {framed}{reason})"


def estimate_section(state: RunState) -> list[str]:
    """The effect estimate, copied from the first tool result, never computed here; or, when
    the design rules an estimate out, the study plan that would settle the question."""
    if state.verdict == "not_causal":
        return []
    if not state.results:
        if state.design is not None and not state.design.estimable:
            return study_plan_section(state)
        return ["## Estimate", "", "No estimate was computed for this question.", ""]
    result = state.results[0]
    r = result.outputs
    lines = backdoor_lines(r) if result.tool == "backdoor_effect" else rct_lines(state, r)
    mapped = "; ".join(f"`{node}` ← `{source}`" for node, source in r["mapped_from"].items())
    lines += ["", f"In the data: {mapped}"]
    if r.get("rows_excluded_missing"):
        lines += ["", f"{r['rows_excluded_missing']} rows with missing values were excluded."]
    return ["## Estimate", "", *lines, "", f"Tool result: `{result.id}`", ""]


def rct_lines(state: RunState, r: dict) -> list[str]:
    """Each arm's risk or mean, then the difference with its CI."""
    q = state.query
    exposure = q.exposure.definition if q else "treatment"
    comparator = q.comparator if q else "control"
    level = f"{r['level']:.0%} CI"
    if r["scale"] == "risk_difference":
        return [
            f"Risk with {exposure}: {r['risk_treated']:.1%} "
            f"({r['events_treated']}/{r['n_treated']}). "
            f"Risk with {comparator}: {r['risk_control']:.1%} "
            f"({r['events_control']}/{r['n_control']}).",
            "",
            f"Risk difference: {r['estimate'] * 100:.1f} percentage points "
            f"({level} {r['ci_low'] * 100:.1f} to {r['ci_high'] * 100:.1f}), "
            f"{r['method']} interval.",
        ]
    return [
        f"With {exposure}: mean {r['mean_treated']:.2f} (n={r['n_treated']}). "
        f"With {comparator}: mean {r['mean_control']:.2f} (n={r['n_control']}).",
        "",
        f"Mean difference: {r['estimate']:.2f} ({level} {r['ci_low']:.2f} to "
        f"{r['ci_high']:.2f}), {r['method']} interval.",
    ]


def backdoor_lines(r: dict) -> list[str]:
    """Adjusted and unadjusted effect side by side, then what was adjusted for and why."""

    def effect(value: float) -> str:
        return f"{value * 100:.1f}" if r["scale"] == "risk_difference" else f"{value:.2f}"

    unit = " percentage points" if r["scale"] == "risk_difference" else ""
    level = f"{r['level']:.0%} CI"
    scale = r["scale"].replace("_", " ")
    if not r["adjusted_for"] and not r.get("precision"):
        return [
            f"Unadjusted {scale}: {effect(r['estimate'])}{unit} "
            f"({level} {effect(r['ci_low'])} to {effect(r['ci_high'])}), {r['method']}.",
            "",
            "Nothing adjusted for: no backdoor path in the DAG and no precision variables.",
            "",
            f"{r['n']} rows used ({r['n_treated']} exposed).",
        ]
    confounders = ", ".join(r["adjusted_for"]) or "none (no backdoor path in the DAG)"
    lines = [
        f"Adjusted {scale}: {effect(r['estimate'])}{unit} "
        f"({level} {effect(r['ci_low'])} to {effect(r['ci_high'])}), {r['method']}.",
        "",
        f"Unadjusted: {effect(r['naive_estimate'])}{unit} "
        f"({level} {effect(r['naive_ci_low'])} to {effect(r['naive_ci_high'])}).",
        "",
        f"Confounders adjusted for: {confounders}.",
    ]
    if r.get("precision"):
        lines.append(f"Also adjusted for, to narrow the CI: {', '.join(r['precision'])}.")
    return [*lines, "", f"{r['n']} rows used ({r['n_treated']} exposed)."]


def study_plan_section(state: RunState) -> list[str]:
    """Why the data cannot give an estimate, and what would settle the question."""
    d = state.design
    assert d is not None
    return [
        "## Study plan",
        "",
        f"**Design found:** {d.design.replace('_', ' ')}. Not estimated: {'; '.join(d.reasons)}.",
        "",
        f"- **Key assumption:** {d.key_assumption}",
        f"- **What it would estimate:** {d.estimand}",
        f"- **Next step:** {d.check}",
        "",
    ]


def data_section(state: RunState) -> list[str]:
    """Provenance of each downloaded dataset: URL, license, hash and retrieval time."""
    if not state.downloads:
        return []
    lines = ["## Data used", ""]
    for d in state.downloads:
        lines.append(
            f"- {d.url} — license {d.license or 'unknown'}, sha256 `{d.sha256}`, "
            f"retrieved {d.retrieved_at}"
        )
    return [*lines, ""]


def evidence_section(state: RunState) -> list[str]:
    """Certainty, evidence-flow counts and the included studies. A study's effect is shown
    only if its claim passed the citation check; otherwise it is marked not verified."""
    if state.verdict == "not_causal" or not (state.rounds or state.studies):
        return []
    flow = evidence_flow(state.studies)
    lines = ["## Evidence", ""]
    if state.certainty is not None:
        reasons = "; ".join(state.certainty.reasons)
        level = state.certainty.level.replace("_", " ")
        lines += [f"**Certainty:** {level} ({reasons})", ""]
    excluded = ", ".join(f"{reason} {n}" for reason, n in flow.excluded.items()) or "none"
    lines += [
        f"Evidence flow: {flow.found} sources found → {flow.after_dedupe} studies after "
        f"dedupe → {flow.included} included (excluded: {excluded}).",
        "",
    ]
    if failed := sum(call.error is not None for call in state.lookups):
        lines += [
            f"Europe PMC lookups failed for {failed} queries (after one retry): the studies "
            "they were for may lack a DOI or PMID and were not checked for retraction.",
            "",
        ]
    included = sorted((s for s in state.studies if s.included), key=lambda s: -design_tier(s))
    if not included:
        lines += ["No included studies. Searches run:", ""]
        lines += [f"- {q.text}" for r in state.rounds for q in r.plan.queries]
        return [*lines, ""]
    verified = {
        c.study_id
        for c in state.claims
        if c.kind in ("effect_estimate", "null_result") and c.status == "documented"
    }
    for s in included:
        effect = s.effect if s.id in verified and s.effect else "effect not verified"
        news = ", known only from secondary sources" if s.secondary_only else ""
        lines.append(
            f"- [{s.title}]({s.url}) — {s.design}, {s.year or 'year unknown'}{news}: {effect}"
        )
    return [*lines, ""]


def byline(study) -> str:
    """`Ruetzler 2013`, or the title when author and year are unknown."""
    parts = [p for p in (study.first_author, study.year and str(study.year)) if p]
    return " ".join(parts) if parts else study.title


def ledger_entries(state: RunState) -> dict[str, str]:
    """What the report model may cite, by id: run facts (`F-…`), tool results and the
    documented claims (quote found and judged to support). Nothing else reaches the prose.

    Study details (design, n, population) are left out: the Evidence and Sources sections
    already show them, and in live D0 they were half the report model's input."""
    entries: dict[str, str] = {}
    if state.verdict is not None:
        entries["F-verdict"] = f"Verdict picked by the controller: {VERDICT_LABELS[state.verdict]}"
    if state.certainty is not None:
        level = state.certainty.level.replace("_", " ")
        entries["F-certainty"] = f"Certainty: {level} ({'; '.join(state.certainty.reasons)})"
    if state.rounds or state.studies:
        flow = evidence_flow(state.studies)
        entries["F-flow"] = (
            f"Evidence flow: {flow.found} sources found, {flow.after_dedupe} studies after "
            f"dedupe, {flow.included} included"
        )
    if state.downloads:
        accepted = next((c for c in state.datasets if c.accepted), None)
        confirmed = "confirmed" if state.mapping_confirmed else "not confirmed by the user"
        title = accepted.title if accepted else state.downloads[-1].url
        rows = f", {state.profile.rows} rows" if state.profile else ""
        entries["F-data"] = f"Data used: {title}{rows}; column mapping {confirmed}"
    elif state.datasets:
        restricted = sum(c.access == "restricted" for c in state.datasets)
        entries["F-data"] = (
            f"No usable dataset: {len(state.datasets)} candidates considered, "
            f"{restricted} in restricted repositories"
        )
    if (d := state.design) is not None:
        entries["F-design"] = (
            f"Data design: {d.design.replace('_', ' ')}; estimable: {d.estimable}; "
            f"key assumption: {d.key_assumption}; estimand: {d.estimand}; next check: "
            f"{d.check}; because: {'; '.join(d.reasons)}"
        )
    if state.verdict == "common_cause":
        entries["F-common-cause"] = (
            "No included trial or observational study tests this link; checked sources name "
            f"causes of both the exposure and the outcome: {', '.join(common_causes(state))}"
        )
    if lacking := unmeasured_confounders(state):
        entries["F-unmeasured"] = (
            f"Confounders named in the literature but not in the data: {', '.join(lacking)}"
        )
    for r in state.results:
        lines = (
            backdoor_lines(r.outputs)
            if r.tool == "backdoor_effect"
            else rct_lines(state, r.outputs)
        )
        entries[r.id] = " ".join(line for line in lines if line)
    studies = {s.id: s for s in state.studies}
    for c in state.claims:
        study = studies.get(c.study_id or "")
        if c.status != "documented" or study is None or not study.included:
            continue
        if c.kind == "study_detail":
            continue
        entries[c.id] = f'{byline(study)}, {study.design} ({c.kind}): {c.text}. Quote: "{c.quote}"'
    return entries


def numbers(text: str) -> set[float]:
    """Every number written in `text`, ignoring sign and thousands commas."""
    return {float(n.replace(",", "")) for n in NUMBER.findall(text)}


def free_numbers(state: RunState) -> set[float]:
    """Numbers any sentence may use without a citation: those in the question and its PECO."""
    text = state.question + (state.query.model_dump_json() if state.query else "")
    return numbers(text)


def sentence_problem(s: Sentence, entries: dict[str, str], free: set[float]) -> str | None:
    """Why the sentence fails the lint, or None if it passes."""
    if not s.cites:
        return "cites nothing; every sentence must cite a ledger entry"
    if unknown := [c for c in s.cites if c not in entries]:
        return f"cites ids not in the ledger: {', '.join(unknown)}"
    text = s.text
    for c in s.cites:
        text = text.replace(c, "")
    allowed = free.union(*(numbers(entries[c]) for c in s.cites))
    if untraced := sorted(numbers(text) - allowed):
        shown = ", ".join(f"{n:g}" for n in untraced)
        return f"numbers {shown} appear in none of its cited entries"
    return None


def lint(draft: ReportDraft, entries: dict[str, str], free: set[float]) -> list[str]:
    """Every problem in the draft, one line each, naming the section and sentence number."""
    problems = []
    for section in ("summary", "comparison", "limitations"):
        for i, s in enumerate(getattr(draft, section), 1):
            if problem := sentence_problem(s, entries, free):
                problems.append(f"{section}[{i}] {s.text!r}: {problem}")
    if (n := sum(len(s.text.split()) for s in draft.summary)) > SUMMARY_WORDS:
        problems.append(f"summary has {n} words; at most {SUMMARY_WORDS}")
    return problems


def keep_passing(draft: ReportDraft, entries: dict[str, str], free: set[float]) -> ReportDraft:
    """The draft minus failing sentences (logged), with the summary cut to its word limit."""

    def passing(sentences: list[Sentence]) -> list[Sentence]:
        kept = []
        for s in sentences:
            if problem := sentence_problem(s, entries, free):
                log.warning("report: dropped sentence %r: %s", s.text, problem)
            else:
                kept.append(s)
        return kept

    summary, words = [], 0
    for s in passing(draft.summary):
        words += len(s.text.split())
        if words > SUMMARY_WORDS:
            log.warning("report: dropped summary sentence over %d words: %r", SUMMARY_WORDS, s.text)
            continue
        summary.append(s)
    return ReportDraft(
        summary=summary,
        comparison=passing(draft.comparison),
        limitations=passing(draft.limitations),
    )


def draft_report(state: RunState, llm: LLMClient) -> ReportDraft:
    """Ask the large model for the prose sections, lint them, allow one rewrite, then drop
    whatever still fails."""
    entries = ledger_entries(state)
    free = free_numbers(state)
    q = state.query
    peco = (
        f"Population: {q.population}\nExposure: {q.exposure.definition}\n"
        f"Comparator: {q.comparator}\nOutcome: {q.outcome.definition}\n"
        f"Time window: {q.time_window or 'not specified'}\n"
        if q
        else ""
    )
    listing = "\n".join(f"[{i}] {text}" for i, text in entries.items())
    user = f"Question: {state.question}\n{peco}\nLedger entries:\n{listing}"
    draft = llm.complete("large", "report", user, ReportDraft)
    if problems := lint(draft, entries, free):
        log.info("report: lint found %d problems; asking for one rewrite", len(problems))
        listed = "\n".join(f"- {p}" for p in problems)
        retry = (
            f"{user}\n\nYour previous draft:\n{draft.model_dump_json()}\n\n"
            f"It failed these checks:\n{listed}\n\nReturn the whole draft again with these "
            "sentences fixed or removed."
        )
        draft = llm.complete("large", "report", retry, ReportDraft)
    if draft.comparison and not state.results:
        # The prompt asks for [] without a result of ours; models still write one and call a
        # published estimate "ours", so the harness enforces it.
        log.warning("report: dropped the comparison, no estimate was computed: %s", draft)
        draft = draft.model_copy(update={"comparison": []})
    return keep_passing(draft, entries, free)
