"""Data step (PRD §5.6, M5): find one public table that covers the question's DAG.

Candidates come from included studies' data links first, then a Tavily search restricted to
allowlisted hosts and the Rdatasets index. A rule filter (no model tokens) rejects restricted,
multi-table, off-allowlist, non-data, oversized, closed-license and off-topic candidates; the
top 10 by keyword match (rare words weigh more) have their header line read, and the small
model says which DAG nodes each covers. The harness computes the scores, then downloads the
best (next on failure, at most 3). If no candidate covers both exposure and outcome, it
searches once more naming what's missing. Every candidate is kept on the run with why it was
rejected or which rank won.
"""

import hashlib
import io
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pandas as pd

from causal_genie.agent.state import (
    DataCoverage,
    DatasetCandidate,
    Download,
    RepositoryFile,
    RunState,
    SourceKind,
)
from causal_genie.llm.client import LLMClient
from causal_genie.tools.derive import DATE_RULES
from causal_genie.tools.download import (
    ALLOWLIST,
    FORMATS,
    GITHUB_REPOS,
    MAX_BYTES,
    DownloadRejected,
    Fetcher,
    allowlisted,
)
from causal_genie.tools.profile import NotSingleTable, profile
from causal_genie.tools.search import Searcher

log = logging.getLogger(__name__)

RDATASETS = "https://vincentarelbundock.github.io/Rdatasets"
RDATASETS_INDEX = f"{RDATASETS}/datasets.csv"
TOP_N = 10
MAX_DOWNLOADS = 3
# Day-of-month column names that, with `year` and `month`, give a calendar date.
DAY_COLUMNS = ("day", "day_of_month", "date_of_month", "mday")

# Data held behind an access application: shown as 🔒, never downloaded.
RESTRICTED_HOSTS = (
    "vivli.org",
    "yoda.yale.edu",
    "biolincc.nhlbi.nih.gov",
    "physionet.org",
    "ukbiobank.ac.uk",
)
# Known multi-table sources (URL fragment → why); a ZIP or workbook with several tables is
# caught after download by `profile`.
MULTI_TABLE = {
    "cdc.gov/nchs/nhanes": "NHANES: separate files joined on SEQN, with survey weights",
}
OPEN_LICENSES = ("cc0", "cc-by", "cc by", "mit", "gpl", "bsd", "apache", "odbl", "pddl", "public")
# Licenses of named sources, checked by hand in the T3 dataset audit (URL prefix → license).
LICENSES = {
    "https://raw.githubusercontent.com/fivethirtyeight/data/": "CC-BY-4.0",
    f"{RDATASETS}/csv/fivethirtyeight/": "CC-BY-4.0",
    "https://raw.githubusercontent.com/higgi13425/medicaldata/": "MIT",
    f"{RDATASETS}/csv/medicaldata/": "MIT",
}
STOPWORDS = {"the", "and", "for", "with", "from", "any", "per", "day", "post", "pre", "number"}

GITHUB_FILE = re.compile(r"https://github\.com/([^/]+)/([^/]+)/(blob|tree)/([^/]+)/?(.*)")
RDATASETS_DOC = re.compile(re.escape(RDATASETS) + r"/doc/([^/]+)/([^/]+)\.html")
PKGDOWN_REFERENCE = re.compile(r"https://[\w-]+\.github\.io/([\w.]+)/reference/([\w.]+)\.html")


def find_data(
    state: RunState,
    llm: LLMClient,
    searcher: Searcher,
    fetcher: Fetcher,
    cache_dir: Path,
    max_downloads: int = MAX_DOWNLOADS,
) -> DatasetCandidate | None:
    """Fill `state.datasets` and return the accepted candidate (downloaded and profiled into
    `state.downloads` / `state.profile`), or None if no usable single table was found."""
    assert state.query is not None, "frame the question before looking for data"
    exposure, outcome = state.query.exposure.name, state.query.outcome.name
    index = read_index(fetcher.download(RDATASETS_INDEX, cache_dir, doi=None, license=None).path)
    kws = keywords([exposure, outcome])

    found = paper_candidates(state, index, fetcher)
    found += search_candidates(
        searcher, f"{words(exposure)} {words(outcome)} dataset", index, fetcher
    )
    found += repository_candidates(index, kws)
    state.datasets = screen(unique(found), state, llm, fetcher, kws, keywords([outcome]), index)

    if not any(c.score is not None and c.rejected_reason is None for c in state.datasets):
        missing = missing_nodes(state.datasets, exposure, outcome)
        log.info("no candidate covers %s and %s; searching once for %s", exposure, outcome, missing)
        seen = {c.url for c in state.datasets}
        query = " ".join(words(n) for n in missing) + " data"
        new = [c for c in search_candidates(searcher, query, index, fetcher) if c.url not in seen]
        state.datasets += screen(unique(new), state, llm, fetcher, kws, keywords(missing), index)

    rank(state.datasets)
    return pick(state, fetcher, cache_dir, max_downloads)


def words(name: str) -> str:
    """A snake_case node name as search words, e.g. `sore_throat` → `sore throat`."""
    return name.replace("_", " ")


def keywords(names: list[str]) -> list[str]:
    """Distinct words of the names worth matching against dataset titles, plurals trimmed."""
    out = []
    for word in re.split(r"[^a-z0-9]+", " ".join(names).lower()):
        if len(word) < 3 or word in STOPWORDS:
            continue
        out.append(word[:-1] if len(word) > 4 and word.endswith("s") else word)
    return list(dict.fromkeys(out))


def mentions(text: str, kw: str) -> bool:
    """`kw` starts a word in `text` (so `birth` matches `US_births`, `throat` not `cutthroat`)."""
    return re.search(rf"(?<![a-z]){re.escape(kw)}", text.lower()) is not None


def keyword_hits(c: DatasetCandidate, kws: list[str]) -> int:
    """How many keywords appear in the candidate's title, URL or description."""
    return sum(mentions(f"{c.title} {c.url} {c.description}", kw) for kw in kws)


def index_text(index: pd.DataFrame) -> pd.Series:
    """Each Rdatasets table's package, name and title, lowercased, for keyword matching."""
    return (index["Package"] + " " + index["Item"] + " " + index["Title"]).str.lower()


def keyword_weights(index: pd.DataFrame, kws: list[str]) -> dict[str, float]:
    """log((1 + N) / (1 + tables whose title has the word)) over the Rdatasets index's N
    tables: a word few dataset titles use ("moon", "birth") counts more than a common one
    ("count", which also starts "country"). A word in no title gets the most, log(1 + N)."""
    text = list(index_text(index))
    return {kw: math.log((1 + len(text)) / (1 + sum(mentions(t, kw) for t in text))) for kw in kws}


def shortlist(
    passing: list[DatasetCandidate], kws: list[str], lead: list[str], index: pd.DataFrame
) -> list[DatasetCandidate]:
    """Candidates in the order their headers are worth reading: paper links first, then by
    weighted hits on the `lead` keywords, then on all `kws`, then larger Rdatasets tables
    first (a tie would otherwise go by name; search results count as 0 rows)."""
    weights = keyword_weights(index, sorted({*kws, *lead}))
    rows = dict(zip(index["CSV"], index["Rows"], strict=True))

    def weighted(c: DatasetCandidate, words: list[str]) -> float:
        text = f"{c.title} {c.url} {c.description}"
        return sum(weights[kw] for kw in words if mentions(text, kw))

    return sorted(
        passing,
        key=lambda c: (
            c.source != "paper",
            -weighted(c, lead),
            -weighted(c, kws),
            -rows.get(c.url, 0),
        ),
    )


def candidate(
    url: str, source: SourceKind, title: str, description: str, size: int | None = None
) -> DatasetCandidate:
    """A not-yet-screened candidate; license only where a named source's is known."""
    license = next((lic for prefix, lic in LICENSES.items() if url.startswith(prefix)), None)
    return DatasetCandidate(
        url=url,
        host=urlsplit(url).hostname or "",
        license=license,
        access="unknown",
        coverage={},
        score=None,
        rejected_reason=None,
        source=source,
        title=title,
        description=description,
        bytes=size,
    )


def read_index(path_or_bytes: str | bytes) -> pd.DataFrame:
    """The Rdatasets index (one row per table), keyed for lookups by (package, item)."""
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    return pd.read_csv(source).set_index(["Package", "Item"], drop=False)


def index_candidate(row: pd.Series, source: SourceKind) -> DatasetCandidate:
    """A candidate for one Rdatasets table."""
    return candidate(
        row["CSV"],
        source,
        f"{row['Package']}::{row['Item']}: {row['Title']}",
        f"{row['Rows']} rows × {row['Cols']} columns (Rdatasets)",
    )


def repository_candidate(
    f: RepositoryFile, source: SourceKind, title: str, description: str
) -> DatasetCandidate:
    """A candidate for one file of a Zenodo, figshare or Dataverse record, carrying the
    record's license and DOI and the format from the file's name."""
    c = candidate(f.url, source, f"{title or f.record_title}: {f.name}", description, f.bytes)
    c.license, c.doi, c.format = f.license, f.doi, f.format
    return c


def resolve_link(
    url: str,
    source: SourceKind,
    title: str,
    description: str,
    index: pd.DataFrame,
    fetcher: Fetcher,
) -> list[DatasetCandidate]:
    """The data files a link points to: an R package's docs page → its Rdatasets CSV, a GitHub
    file → its raw URL, a named GitHub folder → its data files, a Zenodo, figshare or
    Dataverse record page or DOI → its data files. Other links, and a record whose lookup
    failed, are returned as they are, for the rule filter to judge."""
    package_doc = RDATASETS_DOC.fullmatch(url) or PKGDOWN_REFERENCE.fullmatch(url)
    if package_doc and package_doc.groups() in index.index:
        return [index_candidate(index.loc[package_doc.groups()], source)]
    try:
        files = fetcher.record(url)
    except (httpx.HTTPError, DownloadRejected) as e:
        log.warning("record lookup failed for %s: %r", url, e)
        page = candidate(url, source, title, description)
        page.rejected_reason = f"record lookup failed: {e}"
        return [page]
    if files is not None:
        tables = [f for f in files if f.format in FORMATS]
        log.info("record %s: %d of %d files are data files", url, len(tables), len(files))
        return [repository_candidate(f, source, title, description) for f in tables]
    if m := GITHUB_FILE.fullmatch(url):
        owner, repo, kind, ref, path = m.groups()
        if kind == "blob":
            raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
            return [candidate(raw, source, title or path, description)]
        if f"/{owner}/{repo}/" in GITHUB_REPOS:
            return github_folder(owner, repo, ref, path, source, description, fetcher)
    return [candidate(url, source, title, description)]


def github_folder(
    owner: str,
    repo: str,
    ref: str,
    path: str,
    source: SourceKind,
    description: str,
    fetcher: Fetcher,
) -> list[DatasetCandidate]:
    """Data files in a folder of a named GitHub repo, listed by the GitHub contents API."""
    return [
        candidate(f.download_url, source, f"{owner}/{repo}: {f.name}", description, f.size)
        for f in fetcher.listing(owner, repo, ref, path)
        if f.type == "file" and f.download_url and Path(f.name).suffix.lower() in FORMATS
    ]


def paper_candidates(state: RunState, index: pd.DataFrame, fetcher: Fetcher) -> list:
    """Data links that included studies give, resolved to files."""
    found = []
    for s in state.studies:
        if s.included and s.data_link and s.data_link.startswith("https://"):
            title = f"data link from {s.first_author or s.title} {s.year or ''}".strip()
            for c in resolve_link(s.data_link, "paper", title, s.title, index, fetcher):
                found.append(c.model_copy(update={"study_id": s.id}))
    return found


def search_candidates(
    searcher: Searcher, query: str, index: pd.DataFrame, fetcher: Fetcher
) -> list[DatasetCandidate]:
    """Tavily hits restricted to allowlisted hosts, resolved to files."""
    found = []
    for hit in searcher.search(query, domains=tuple(ALLOWLIST), mode="restrict"):
        found += resolve_link(hit.url, "search", hit.title, hit.content[:300], index, fetcher)
    return found


def repository_candidates(index: pd.DataFrame, kws: list[str]) -> list[DatasetCandidate]:
    """Rdatasets tables whose package, name or title contains a keyword."""
    hits = index_text(index).apply(lambda t: any(mentions(t, kw) for kw in kws))
    return [index_candidate(row, "repository") for _, row in index[hits].iterrows()]


def unique(found: list[DatasetCandidate]) -> list[DatasetCandidate]:
    """First candidate per URL, so a paper's link outranks the same file found by search."""
    by_url: dict[str, DatasetCandidate] = {}
    for c in found:
        by_url.setdefault(c.url, c)
    return list(by_url.values())


def rule_filter(c: DatasetCandidate, kws: list[str]) -> None:
    """Set `access` and, if a rule fails, `rejected_reason` (PRD §5.6); no model tokens. A
    reason set earlier (a failed record lookup) is kept."""
    host = c.host
    restricted = next((h for h in RESTRICTED_HOSTS if host == h or host.endswith(f".{h}")), None)
    multi = next((why for frag, why in MULTI_TABLE.items() if frag in c.url), None)
    suffix = c.format or Path(urlsplit(c.url).path).suffix.lower()
    c.access = "restricted" if restricted else "open" if allowlisted(c.url) else "unknown"
    if c.rejected_reason is not None:  # e.g. a failed record lookup: keep why
        return
    if restricted:
        c.rejected_reason = f"restricted access ({restricted}): apply to the data holder"
    elif multi:
        c.rejected_reason = f"multi-table dataset, not usable in the MVP ({multi})"
    elif not allowlisted(c.url):
        c.rejected_reason = "not on the allowlist"
    elif suffix not in FORMATS:
        c.rejected_reason = f"not a data file (format {suffix or 'none'}); a landing page?"
    elif c.bytes is not None and c.bytes > MAX_BYTES:
        c.rejected_reason = f"size {c.bytes} bytes over the {MAX_BYTES} cap"
    elif c.license and not any(lic in c.license.lower() for lic in OPEN_LICENSES):
        c.rejected_reason = f"license not open: {c.license}"
    elif c.source != "paper" and keyword_hits(c, kws) == 0:
        c.rejected_reason = f"no keyword ({', '.join(kws)}) in title or URL"


def screen(
    found: list[DatasetCandidate],
    state: RunState,
    llm: LLMClient,
    fetcher: Fetcher,
    kws: list[str],
    lead: list[str],
    index: pd.DataFrame,
) -> list[DatasetCandidate]:
    """Rule-filter the candidates, read the top 10's headers, have the model score them.

    The top 10 come from `shortlist`; `lead` is the outcome's keywords, since a table must
    hold the outcome while an exposure like moon phase may be derived.
    """
    for c in found:
        rule_filter(c, kws)
    passing = shortlist([c for c in found if c.rejected_reason is None], kws, lead, index)
    for c in passing[TOP_N:]:
        c.rejected_reason = f"outside the top {TOP_N} by keyword match"
    top = passing[:TOP_N]
    for c in top:
        try:
            c.columns = fetcher.peek(c.url, c.format)
        except (DownloadRejected, httpx.HTTPError) as e:
            log.warning("peek failed for %s: %r", c.url, e)
            c.rejected_reason = f"header could not be read: {e}"
    scorable = [c for c in top if c.rejected_reason is None]
    if scorable:
        score(scorable, state, llm)
    log.info(
        "screened %d candidates: %d passed the rules, %d scored",
        len(found),
        len(passing),
        len(scorable),
    )
    return found


def dag_nodes(state: RunState) -> list[str]:
    """Exposure, outcome, then every other DAG node, in a stable order."""
    assert state.query is not None
    first = [state.query.exposure.name, state.query.outcome.name]
    others = sorted({n for e in state.edges for n in (e.source, e.target)} - set(first))
    return first + others


def score(cands: list[DatasetCandidate], state: RunState, llm: LLMClient) -> None:
    """Ask the small model which DAG nodes each candidate covers; the harness turns that into
    a score (share of nodes covered) and rejects tables lacking the exposure or outcome."""
    assert state.query is not None
    nodes = dag_nodes(state)
    listing = "\n\n".join(
        f"{i}. {c.title}\n   URL: {c.url}\n   About: {c.description}\n"
        f"   Columns: {', '.join(c.columns) if c.columns else '(not known)'}"
        for i, c in enumerate(cands, 1)
    )
    user = (
        f"Exposure: {state.query.exposure.name} ({state.query.exposure.definition})\n"
        f"Outcome: {state.query.outcome.name} ({state.query.outcome.definition})\n"
        f"Population: {state.query.population}\n"
        f"DAG nodes: {', '.join(nodes)}\n\n"
        f"Candidates:\n\n{listing}"
    )
    answer = llm.complete("small", "data_coverage", user, DataCoverage)
    by_number = {a.candidate: a for a in answer.candidates}
    for i, c in enumerate(cands, 1):
        if i not in by_number:
            c.rejected_reason = "not scored by the model"
            continue
        given = {m.node: m.column for m in by_number[i].coverage}
        if unknown := set(given) - set(nodes):
            log.warning("candidate %d: ignored coverage of unknown nodes %s", i, sorted(unknown))
        c.coverage = {n: given.get(n) for n in nodes}
        cover_calendar_nodes(c)
        c.score = sum(col is not None for col in c.coverage.values()) / len(nodes)
        lacks = [n for n in nodes[:2] if c.coverage[n] is None]
        if lacks:
            c.rejected_reason = f"lacks {', '.join(lacks)}"


def calendar_columns(columns: list[str]) -> str | None:
    """The column(s) a calendar date can be built from, as `derive.to_dates` takes them: one
    date column (`date`, `*_date`), or year, month and day columns; None if there are none."""
    lower = {col.lower(): col for col in columns}
    for col in columns:
        if col.lower() == "date" or col.lower().endswith("_date"):
            return col
    day = next((lower[d] for d in DAY_COLUMNS if d in lower), None)
    if "year" in lower and "month" in lower and day:
        return f"{lower['year']}, {lower['month']}, {day}"
    return None


def cover_calendar_nodes(c: DatasetCandidate) -> None:
    """Mark a node named like a `derive` date rule (e.g. `full_moon`) covered when the table has
    a date, whatever the model said: the harness can compute it, and the small model was
    inconsistent about this. A column the model gave is kept."""
    if not (dates := calendar_columns(c.columns)):
        return
    for node, col in c.coverage.items():
        if node in DATE_RULES and col is None:
            c.coverage[node] = dates
            log.info("candidate %s: %s covered from %s by the harness", c.url, node, dates)


def missing_nodes(cands: list[DatasetCandidate], exposure: str, outcome: str) -> list[str]:
    """Exposure/outcome that the best-scored candidate lacks; both if nothing was scored."""
    scored = [c for c in cands if c.score is not None]
    if not scored:
        return [exposure, outcome]
    best = max(scored, key=lambda c: c.score or 0)
    return [n for n in (exposure, outcome) if best.coverage.get(n) is None]


def rank(cands: list[DatasetCandidate]) -> None:
    """Rank scored candidates 1..n: usable ones first, then by score (stable on ties)."""
    scored = [c for c in cands if c.score is not None]
    scored.sort(key=lambda c: (c.rejected_reason is not None, -(c.score or 0)))
    for i, c in enumerate(scored, 1):
        c.rank = i


def pick(
    state: RunState, fetcher: Fetcher, cache_dir: Path, max_downloads: int
) -> DatasetCandidate | None:
    """Download and profile usable candidates in rank order until one is a single table."""
    usable = sorted(
        (c for c in state.datasets if c.rank is not None and c.rejected_reason is None),
        key=lambda c: c.rank or 0,
    )
    for c in usable[:max_downloads]:
        try:
            d = fetcher.download(c.url, cache_dir, doi=c.doi, license=c.license, suffix=c.format)
            table = profile(Path(d.path))
        except (DownloadRejected, NotSingleTable, pd.errors.ParserError, httpx.HTTPError) as e:
            log.warning("candidate rank %s failed: %s", c.rank, e)
            c.rejected_reason = f"download failed: {e}"
            continue
        c.accepted = True
        state.downloads.append(d)
        state.profile = table
        log.info("accepted dataset rank=%d url=%s score=%.2f", c.rank, c.url, c.score)
        return c
    log.info("no usable dataset among %d candidates", len(state.datasets))
    return None


def use_table(state: RunState, path: Path) -> DatasetCandidate:
    """Accept the user's own table (`--data`) without searching: profile it, record its
    checksum, and make it the run's only candidate. Rows still never reach a model."""
    body = path.read_bytes()
    state.profile = profile(path)
    state.downloads.append(
        Download(
            url=str(path),
            path=str(path),
            sha256=hashlib.sha256(body).hexdigest(),
            bytes=len(body),
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            doi=None,
            license=None,
        )
    )
    accepted = candidate(str(path), "user", path.name, "table given by the user", len(body))
    accepted.access, accepted.accepted, accepted.rank = "open", True, 1
    accepted.columns = [c.name for c in state.profile.columns]
    accepted.description = f"{state.profile.rows} rows × {len(accepted.columns)} columns"
    state.datasets = [accepted]
    log.info("using the user's table %s (%s)", path, accepted.description)
    return accepted
