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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import ValidationError

from causal_genie.agent.router import normalize
from causal_genie.agent.state import RunState, Schema, Sentence, Verdict
from causal_genie.agent.steps.report import (
    NUMBER,
    VERDICT_LABELS,
    byline,
    dag_section,
    numbers,
)
from causal_genie.llm.client import LLMClient, ModelOutputError

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
    "common_cause": (
        "No study tests this link directly. The sources we found explain it by something that "
        "drives both (see the graph), not by one causing the other."
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


LABEL = "label.txt"  # a gallery run's short tag, written by hand, e.g. "full A/B data"


def _gallery(runs_dir: Path) -> dict[str, tuple[RunState, str]]:
    """Picker label → (run, its `label.txt` tag) for each run the gallery offers.

    Runs of one question are told apart by their `label.txt` (the demos ask the same question
    of different data); without one, the latest run of the question stands for it, as in
    `saved_runs`.
    """
    runs: dict[str, tuple[RunState, str]] = {}
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
        if not state.report:
            continue
        tag = path.parent / LABEL
        label = tag.read_text().strip() if tag.is_file() else ""
        runs[f"{state.question} · {label}" if label else state.question] = (state, label)
    return runs


def gallery_runs(runs_dir: Path) -> dict[str, tuple[str, Path]]:
    """Picker label → (question, `state.json`), one entry per run the gallery offers."""
    return {
        key: (state.question, runs_dir / state.run_id / "state.json")
        for key, (state, _) in _gallery(runs_dir).items()
    }


PARTIAL = "Partial run: no verdict reached"  # a card's verdict when the run stopped early
ANSWER_CHARS, GENIE_CHARS = 220, 160  # what fits on a card; the full text is one click away
PER_PAGE = 3  # cards in a band's row; the arrows page through the rest


@dataclass
class Card:
    """A gallery run at a glance. Model text is already escaped for `st.markdown`."""

    run_id: str
    question: str
    tag: str  # the run's `label.txt`, e.g. "full A/B data", or ""
    verdict: str  # the verdict's label, or why there is none
    answer: str  # the summary's first sentence, which states the answer
    genie: str  # the Genie's aside, or ""


def clip(text: str, limit: int) -> str:
    """`text` cut at a word before `limit` characters, with an ellipsis if anything was cut."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:.") + "…"


def card(state: RunState, tag: str = "") -> Card:
    if state.failure is None and state.verdict is not None:
        verdict = VERDICT_LABELS[state.verdict]
    else:
        verdict = PARTIAL
    summary = state.prose.summary if state.prose is not None else []
    answer = escape(clip(summary[0].text, ANSWER_CHARS)) if summary else ""
    genie = escape(clip(state.genie, GENIE_CHARS)) if state.genie else ""
    return Card(state.run_id, state.question, tag, verdict, answer, genie)


def gallery_cards(runs_dir: Path) -> list[Card]:
    """A card for each run the gallery offers, oldest first."""
    return [card(state, tag) for state, tag in _gallery(runs_dir).values()]


def matching(cards: list[Card], query: str) -> list[Card]:
    """The cards whose question or tag has every word of `query`, ignoring case and order."""
    words = query.lower().split()
    return [c for c in cards if all(w in f"{c.question} {c.tag}".lower() for w in words)]


def verdict_choices(cards: list[Card]) -> list[str]:
    """The verdicts the cards have, in the usual verdict order, partial runs last."""
    have = {c.verdict for c in cards}
    return [v for v in [*VERDICT_LABELS.values(), PARTIAL] if v in have]


def with_verdicts(cards: list[Card], verdicts: list[str]) -> list[Card]:
    """The cards with one of `verdicts`; all of them when none is picked."""
    return [c for c in cards if not verdicts or c.verdict in verdicts]


# Gallery bands, in the order they are shown, and the questions in each (any case and
# punctuation). A question not listed here goes in the last band, `OTHER`.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "🏠 Daily life": (
        "Does listening to music while studying improve exam scores?",
        "Does shaving make hair grow back thicker?",
        "Does cracking your knuckles cause arthritis?",
    ),
    "🩺 Health": (
        "Does licorice gargling reduce post-op sore throat?",
        "Does sugar cause hyperactivity in children?",
        "Do carrots improve night vision?",
    ),
    "🎈 Silly & fun": (
        "Does the full moon cause more births?",
        "Do ice cream sales at beaches cause shark attacks to occur?",
    ),
    "📈 Business (A/B tests)": ("Did the new onboarding flow raise 7-day activation?",),
}
OTHER = "🗂️ Miscellaneous"
_CATEGORY_OF = {normalize(q): name for name, qs in CATEGORIES.items() for q in qs}

Topic = Literal["daily_life", "health", "fun", "business", "misc"]
TOPICS: dict[Topic, str] = {
    "daily_life": "🏠 Daily life",
    "health": "🩺 Health",
    "fun": "🎈 Silly & fun",
    "business": "📈 Business (A/B tests)",
    "misc": OTHER,
}


class TopicLabel(Schema):
    """The model's pick of a gallery band for a question off the `CATEGORIES` list."""

    topic: Topic


def listed(question: str) -> str:
    """The question's band from the `CATEGORIES` list, or `OTHER`."""
    return _CATEGORY_OF.get(normalize(question), OTHER)


def category(question: str, llm: LLMClient | None) -> str:
    """The question's gallery band: from the `CATEGORIES` list if it is there, else the small
    model's pick. Without a model, or if its answer fails the schema twice, `OTHER`: a band
    is only where a card is shown, so a failed pick never stops the gallery."""
    if (band := listed(question)) != OTHER or llm is None:
        return band
    try:
        label = llm.complete("small", "topic", f"<question>\n{question}\n</question>", TopicLabel)
    except ModelOutputError:
        log.exception("topic: no valid pick for %r", question)
        return OTHER
    log.info("topic: %r → %s", question, label.topic)
    return TOPICS[label.topic]


def bands(
    cards: list[Card], category_of: Callable[[str], str] = listed
) -> list[tuple[str, list[Card]]]:
    """The cards grouped by `category_of` their question, in `CATEGORIES` order with `OTHER`
    last; no empty band."""
    grouped: dict[str, list[Card]] = {name: [] for name in [*CATEGORIES, OTHER]}
    for c in cards:
        grouped[category_of(c.question)].append(c)
    return [(name, cs) for name, cs in grouped.items() if cs]


def verdict_guide(picked: Verdict | None) -> str:
    """A Markdown table of every verdict and what it means, marking the one this run got."""
    rows = ["| Verdict | What it means |", "|---|---|"]
    for verdict, label in VERDICT_LABELS.items():
        mark = " **← this run**" if verdict == picked else ""
        rows.append(f"| {label}{mark} | {VERDICT_MEANINGS[verdict]} |")
    return "\n".join(rows)


def verdict_list() -> str:
    """Every verdict's name, each opening on a click to what it means (HTML `<details>`, since
    Streamlit can't nest expanders). Labels and meanings are the app's own text."""
    return "\n".join(
        f"<details><summary>{label}</summary>{VERDICT_MEANINGS[verdict]}</details>"
        for verdict, label in VERDICT_LABELS.items()
    )


def cited_ids(state: RunState) -> list[str]:
    """Every id the prose cites, in order of first citation, then the claims backing the
    graph's arrows, so a study that backs only an arrow still gets a number and a card."""
    prose = state.prose
    sentences = prose.summary + prose.comparison + prose.limitations if prose else []
    in_prose = [c for s in sentences for c in s.cites]
    return list(dict.fromkeys([*in_prose, *(c for e in state.edges for c in e.claim_ids)]))


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


def dag_backing(state: RunState) -> str:
    """Markdown list of each arrow with what backs it, the studies as links to their source
    cards (`[Study 1]`), then the report's note on exogenous nodes. Node names are model text."""
    cites = citations(state)
    lines = []
    for e in state.edges:
        if e.question:
            backing = "the question being tested"
        elif e.assumed:
            backing = "assumed (no source)"
        else:
            refs = [cites.get(c, c) for c in e.claim_ids]
            unique = {r.key if isinstance(r, Citation) else r: r for r in refs}.values()
            backing = " ".join(
                f'[<a href="#{anchor(r.key)}">{r.label}</a>]'
                if isinstance(r, Citation)
                else escape(r)
                for r in unique
            )
        lines.append(f"- {escape(e.source)} → {escape(e.target)}: {backing}")
    note = [line for line in dag_section(state) if line.startswith("Treated as exogenous")]
    return "\n".join(lines + ["", *note] if note else lines)


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
