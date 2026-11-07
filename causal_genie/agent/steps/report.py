"""Plain Markdown report built only from run state: the verdict, tool results and sources.

No model writes here, so every number is copied from a `Result`. The model-written report with
lint (T10.2–T10.3) replaces the prose later; the rule that numbers come only from results stays.
"""

from causal_genie.agent.certainty import evidence_flow
from causal_genie.agent.state import RunState, Verdict

VERDICT_LABELS: dict[Verdict, str] = {
    "estimated": "✅ Estimated from data",
    "estimated_strong_assumptions": "⚠️ Estimated, strong assumptions",
    "published_only": "📚 Published evidence only",
    "restricted": "🔒 Data restricted",
    "study_plan": "🧪 Not estimated — study plan",
    "insufficient": "❓ Insufficient evidence",
    "not_causal": "🚫 Not causal",
}

DISCLAIMER = (
    "Causal Genie writes research summaries, not medical advice. Check with a qualified "
    "professional before acting on them."
)


def write_report(state: RunState) -> str:
    """Render the run as Markdown; the controller must have set the verdict first."""
    assert state.verdict is not None, "the controller sets the verdict before the report"
    lines = [f"# {state.question}", "", f"**Verdict:** {VERDICT_LABELS[state.verdict]}", ""]
    if state.verdict == "not_causal" and state.route is not None:
        lines += [state.route.reason, ""]
        if state.route.causal_rewrite:
            lines += [f"A causal question you might mean: *{state.route.causal_rewrite}*", ""]
    lines += question_section(state) + estimate_section(state) + data_section(state)
    lines += evidence_section(state)
    lines += ["---", "", DISCLAIMER, "", f"Run ID: `{state.run_id}`", ""]
    return "\n".join(lines)


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
        f"- **Estimand:** {q.target}, {q.effect_scale.replace('_', ' ')}, "
        f"deviations handled {q.deviation_strategy.replace('_', ' ')}",
    ]
    if q.ambiguities:
        lines.append(f"- **Unresolved ambiguities:** {'; '.join(q.ambiguities)}")
    return [*lines, ""]


def estimate_section(state: RunState) -> list[str]:
    """The effect estimate, copied from the first tool result, never computed here."""
    if state.verdict == "not_causal":
        return []
    if not state.results:
        return ["## Estimate", "", "No estimate was computed for this question.", ""]
    q, r = state.query, state.results[0].outputs
    exposure = q.exposure.definition if q else "treatment"
    comparator = q.comparator if q else "control"
    level = f"{r['level']:.0%} CI"
    if r["scale"] == "risk_difference":
        lines = [
            f"Risk with {exposure}: {r['risk_treated']:.1%} "
            f"({r['events_treated']}/{r['n_treated']}). "
            f"Risk with {comparator}: {r['risk_control']:.1%} "
            f"({r['events_control']}/{r['n_control']}).",
            "",
            f"Risk difference: {r['estimate'] * 100:.1f} percentage points "
            f"({level} {r['ci_low'] * 100:.1f} to {r['ci_high'] * 100:.1f}), "
            f"{r['method']} interval.",
        ]
    else:
        lines = [
            f"With {exposure}: mean {r['mean_treated']:.2f} (n={r['n_treated']}). "
            f"With {comparator}: mean {r['mean_control']:.2f} (n={r['n_control']}).",
            "",
            f"Mean difference: {r['estimate']:.2f} ({level} {r['ci_low']:.2f} to "
            f"{r['ci_high']:.2f}), {r['method']} interval.",
        ]
    cutoff = "" if r["cutoff"] is None else f" > {r['cutoff']:g}"
    lines += [
        "",
        f"Treatment in the data: `{r['treatment_column']}`. "
        f"Outcome in the data: `{r['outcome_column']}`{cutoff}",
    ]
    if r.get("rows_excluded_missing"):
        lines += ["", f"{r['rows_excluded_missing']} rows with missing values were excluded."]
    return ["## Estimate", "", *lines, "", f"Tool result: `{state.results[0].id}`", ""]


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
    included = [s for s in state.studies if s.included]
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
