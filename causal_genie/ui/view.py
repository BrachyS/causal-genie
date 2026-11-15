"""What the Streamlit app shows, as plain functions so it can be tested without a browser.

The app only displays saved runs: every number on screen is copied from run state, and each
number in model-written prose links to the ledger entry (tool result or quoted claim) that
backs it, by the same rule the report lint uses. Prose, quotes and titles come from a model
or the web, so they are escaped before any link is added.
"""

import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from pydantic import ValidationError

from causal_genie.agent.state import RunState, Sentence, Verdict
from causal_genie.agent.steps.report import NUMBER, VERDICT_LABELS, byline, numbers

log = logging.getLogger(__name__)

MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]()#+!|~])")


VERDICT_MEANINGS: dict[Verdict, str] = {
    "estimated": "We found real data on the question and computed the effect ourselves.",
    "estimated_strong_assumptions": (
        "We computed the effect from real data, but the data lack a factor the studies say "
        "matters, so the number leans on assumptions."
    ),
    "published_only": (
        "We found and checked published studies, but no data we could analyze ourselves; "
        "the answer rests on what those studies report."
    ),
    "restricted": (
        "The data that could answer this exist but need an application to access, so we "
        "could not analyze them."
    ),
    "study_plan": (
        "The data we found can't answer this fairly, so instead of a number we describe "
        "the study that could."
    ),
    "insufficient": "The studies we found were too few or too conflicting to say either way.",
    "not_causal": "This asks what is, not what causes what, so there is no effect to estimate.",
}


PLURAL = {"Study": "Studies", "Analysis": "Analyses"}


@dataclass
class Citation:
    key: str  # the card it links to: a study id or a tool result id
    kind: str  # "Study" or "Analysis"
    n: int

    @property
    def label(self) -> str:
        return f"{self.kind} {self.n}"


@dataclass
class SourceCard:
    key: str  # what the prose links to (see `Citation.key`)
    label: str  # e.g. "Study 1"
    markdown: str


def saved_runs(runs_dir: Path) -> dict[str, Path]:
    """Question → `state.json` of its latest run that wrote a report (partial ones included).

    A run saved under an older schema can't be shown, so it is left out (logged) rather than
    offered and then failing when picked.
    """
    runs: dict[str, Path] = {}
    for path in sorted(runs_dir.glob("*/state.json")):  # run ids sort by time
        try:
            state = RunState.load(path)
        except ValidationError as e:
            log.warning(
                "run %s skipped: not loadable under the current schema (%d errors)",
                path.parent.name,
                e.error_count(),
            )
            continue
        if state.report:
            runs[state.question] = path
    return runs


def verdict_guide(picked: Verdict | None) -> str:
    """A Markdown table of every verdict and what it means, marking the one this run got."""
    rows = ["| Verdict | What it means |", "|---|---|"]
    for verdict, label in VERDICT_LABELS.items():
        mark = " **← this run**" if verdict == picked else ""
        rows.append(f"| {label}{mark} | {VERDICT_MEANINGS[verdict]} |")
    return "\n".join(rows)


def cited_ids(state: RunState) -> list[str]:
    """Every id the prose cites, in order of first citation."""
    prose = state.prose
    if prose is None:
        return []
    sentences = prose.summary + prose.comparison + prose.limitations
    return list(dict.fromkeys(c for s in sentences for c in s.cites))


def citations(state: RunState) -> dict[str, Citation]:
    """How each cited id is shown: claims under their study's number (`Study 1`), tool
    results as `Analysis 1`, numbered in citation order. Ledger ids mean nothing to a
    reader; run facts (`F-…`) and unknown ids are left out."""
    claims = {c.id: c for c in state.claims}
    studies = {s.id for s in state.studies}
    results = {r.id for r in state.results}
    keys: dict[str, tuple[str, str]] = {}
    for ref in cited_ids(state):
        if ref in results:
            keys[ref] = (ref, "Analysis")
        elif ref in claims and (study_id := claims[ref].study_id) in studies:
            keys[ref] = (study_id, "Study")
    numbers: dict[str, int] = {}
    counts = {"Study": 0, "Analysis": 0}
    for key, kind in keys.values():
        if key not in numbers:
            counts[kind] += 1
            numbers[key] = counts[kind]
    return {ref: Citation(key, kind, numbers[key]) for ref, (key, kind) in keys.items()}


def anchor(entry_id: str) -> str:
    """HTML id of a ledger entry's source card."""
    return "cite-" + re.sub(r"[^A-Za-z0-9_-]", "-", entry_id)


def escape(text: str) -> str:
    """Model or web text as inert Markdown: no HTML, no links, no emphasis."""
    return MARKDOWN_SPECIAL.sub(r"\\\1", html.escape(text, quote=False))


def cited_html(s: Sentence, entries: dict[str, str], cites: dict[str, Citation]) -> str:
    """The sentence with each number linked to the first cited entry that contains it, then
    its citations as links: `[Study 1]`, `[Studies 1, 3]`, `[Analysis 1]`."""
    cited = [c for c in s.cites if c in entries and c in cites]
    parts, end = [], 0
    for m in NUMBER.finditer(s.text):
        value = float(m.group().replace(",", ""))
        backing = next((c for c in cited if value in numbers(entries[c])), None)
        parts.append(escape(s.text[end : m.start()]))
        number = escape(m.group())
        link = f'<a href="#{anchor(cites[backing].key)}">{number}</a>' if backing else number
        parts.append(link)
        end = m.end()
    parts.append(escape(s.text[end:]))
    shown = sorted({(cites[c].kind, cites[c].n): cites[c] for c in cited}.items())
    brackets = []
    for kind in ("Study", "Analysis"):
        group = [c for (k, _), c in shown if k == kind]
        if len(group) == 1:
            brackets.append(f'[<a href="#{anchor(group[0].key)}">{group[0].label}</a>]')
        elif group:
            links = ", ".join(f'<a href="#{anchor(c.key)}">{c.n}</a>' for c in group)
            brackets.append(f"[{PLURAL[kind]} {links}]")
    return "".join(parts) + "".join(f" {b}" for b in brackets)


def text_fragment(url: str, quote_text: str) -> str:
    """`url` scrolled to the quote's first passage (browser text fragment)."""
    first = quote_text.split("...")[0].strip()
    return f"{url}#:~:text={quote(first)}" if first else url


def sources(state: RunState) -> list[SourceCard]:
    """A card for each study or tool result the prose cites, in citation order: the study's
    cited claims with their verbatim quotes linked to the page, or the tool's name, version
    and outputs."""
    cites = citations(state)
    claims = {c.id: c for c in state.claims}
    studies = {s.id: s for s in state.studies}
    results = {r.id: r for r in state.results}
    cards: dict[str, SourceCard] = {}
    for ref, cite in cites.items():
        if ref in results:
            r = results[ref]
            outputs = json.dumps(r.outputs, indent=2, default=str)
            md = f"**Tool `{r.tool}` {r.tool_version}** (inputs hash `{r.inputs_hash}`)\n\n"
            cards[cite.key] = SourceCard(cite.key, cite.label, md + f"```json\n{outputs}\n```")
            continue
        study, c = studies[cite.key], claims[ref]
        if cite.key not in cards:
            md = f"**{escape(byline(study))}** · {escape(study.design)}"
            cards[cite.key] = SourceCard(cite.key, cite.label, md)
        link = text_fragment(study.url, c.quote or "")
        cards[
            cite.key
        ].markdown += f"\n\n> [“{escape(c.quote or '')}”]({link})\n\n{escape(c.text)} ({c.status})"
    return list(cards.values())
