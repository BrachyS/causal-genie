import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import httpx
import pytest
from tavily import TavilyClient

from causal_genie.agent.state import DatasetCandidate, RunState
from causal_genie.agent.steps import data
from causal_genie.agent.steps.data import (
    candidate,
    find_data,
    keyword_hits,
    keywords,
    resolve_link,
    rule_filter,
)
from causal_genie.tools import download as dl
from causal_genie.tools.search import Searcher

RD = "https://vincentarelbundock.github.io/Rdatasets"
LICORICE = f"{RD}/csv/medicaldata/licorice_gargle.csv"
LICORICE_COPY = f"{RD}/csv/medicaldata/licorice_copy.csv"
BIRTHS = f"{RD}/csv/fivethirtyeight/US_births_2000_2014.csv"
TRIAL_CSV = b"rownames,treat,pacu30min_throatPain,preOp_smoking\n1,0,3,1\n2,1,0,0\n"
ROW = (
    '"{pkg}","{item}","{title}",235,19,3,0,0,0,19,'
    '"{rd}/csv/{pkg}/{item}.csv","{rd}/doc/{pkg}/{item}.html"'
)
INDEX_HEADER = (
    '"Package","Item","Title","Rows","Cols","n_binary","n_character","n_factor",'
    '"n_logical","n_numeric","CSV","Doc"'
)


def index_csv(*rows: tuple[str, str, str]) -> bytes:
    """A small Rdatasets index with the given (package, item, title) rows."""
    lines = [ROW.format(pkg=p, item=i, title=t, rd=RD) for p, i, t in rows]
    return "\n".join([INDEX_HEADER, *lines]).encode()


INDEX = index_csv(
    ("medicaldata", "licorice_gargle", "Randomized Trial of Licorice Gargle before Intubation"),
    ("medicaldata", "licorice_copy", "Licorice gargle trial, second export"),
    ("fivethirtyeight", "US_births_2000_2014", "Some People Are Too Superstitious"),
    ("datasets", "mtcars", "Motor Trend Car Road Tests"),
)
GITHUB_LISTING = [
    {
        "name": "README.md",
        "type": "file",
        "size": 826,
        "download_url": "https://raw.githubusercontent.com/fivethirtyeight/data/master/births/README.md",
    },
    {
        "name": "US_births_2000-2014_SSA.csv",
        "type": "file",
        "size": 96748,
        "download_url": "https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_2000-2014_SSA.csv",
    },
]


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "resolve_ips", lambda host: ["185.199.108.153"])


@pytest.fixture
def served() -> dict[str, bytes | int]:
    """URL → body (or HTTP status) served by the fake web; tests add or change entries."""
    return {
        data.RDATASETS_INDEX: INDEX,
        LICORICE: TRIAL_CSV,
        LICORICE_COPY: TRIAL_CSV,
        BIRTHS: b"rownames,year,month,date_of_month,day_of_week,births\n1,2000,1,1,6,9083\n",
    }


@pytest.fixture
def http(served) -> tuple[httpx.Client, list[str]]:
    """Fake web serving `served`, plus the list of URLs requested."""
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        if url.startswith("https://api.github.com/repos/fivethirtyeight/data/contents/births"):
            return httpx.Response(200, content=json.dumps(GITHUB_LISTING))
        body = served.get(url, 404)
        return httpx.Response(body) if isinstance(body, int) else httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handle)), seen


def coverage(*per_candidate: Mapping[str, str | None]) -> str:
    """Score-data model output: candidate i (1-based) covers the given node → column map."""
    return json.dumps(
        {
            "candidates": [
                {"candidate": i, "coverage": [{"node": n, "column": c} for n, c in cov.items()]}
                for i, cov in enumerate(per_candidate, 1)
            ]
        }
    )


FULL = {"licorice_gargle": "treat", "sore_throat": "pacu30min_throatPain"}


@pytest.fixture
def run_find(make_llm, fake_tavily, http, sample_state: RunState, tmp_path: Path):
    """Run `find_data` on the D1 sample state with fakes, given the model outputs in order."""

    def run(outcomes: list[str]):
        llm, fake = make_llm(outcomes)
        client, _ = http
        found = find_data(
            sample_state, llm, Searcher(cast(TavilyClient, fake_tavily)), client, tmp_path
        )
        return found, sample_state, fake

    return run


def test_keywords_come_from_exposure_and_outcome_names() -> None:
    assert keywords(["full_moon", "births"]) == ["full", "moon", "birth"]
    assert keywords(["licorice_gargle", "post_op_sore_throat"]) == [
        "licorice",
        "gargle",
        "sore",
        "throat",
    ]


def make(url: str, **changes) -> DatasetCandidate:
    """A search candidate titled 'licorice gargle data' unless changed."""
    c = candidate(url, "search", "licorice gargle data", "")
    for field, value in changes.items():
        setattr(c, field, value)
    return c


KW = ["licorice", "gargle"]


@pytest.mark.parametrize(
    ("c", "reason"),
    [
        (make("https://www.vivli.org/study/123"), "restricted"),
        (make("https://physionet.org/content/mimiciv/2.2/"), "restricted"),
        (make("https://wwwn.cdc.gov/nchs/nhanes/Default.aspx"), "not usable in the MVP"),
        (make("https://example.com/licorice.csv"), "allowlist"),
        (make("https://zenodo.org/records/123"), "format"),
        (make(f"{RD}/csv/x/big.csv", bytes=60 * 1024 * 1024), "size"),
        (make(LICORICE, license="All rights reserved"), "license"),
        (make(f"{RD}/csv/datasets/mtcars.csv", title="Motor Trend Car Road Tests"), "keyword"),
    ],
)
def test_rule_filter_rejects_with_reason(c: DatasetCandidate, reason: str) -> None:
    rule_filter(c, KW)

    assert c.rejected_reason is not None and reason in c.rejected_reason


def test_restricted_host_is_marked_restricted() -> None:
    c = make("https://yoda.yale.edu/trials/NCT1")

    rule_filter(c, KW)

    assert c.access == "restricted"


@pytest.mark.parametrize(
    "c",
    [
        make(LICORICE, license="MIT"),
        make(LICORICE, license=None),  # unknown license is kept; the report says so
        make(LICORICE, source="paper", title="data link from Ruetzler 2013"),  # no keyword needed
    ],
)
def test_rule_filter_passes_open_allowlisted_tables(c: DatasetCandidate) -> None:
    rule_filter(c, KW)

    assert c.rejected_reason is None
    assert c.access == "open"


def test_resolve_github_blob_to_raw_file(http) -> None:
    client, _ = http
    url = "https://github.com/fivethirtyeight/data/blob/master/births/US_births_2000-2014_SSA.csv"

    (c,) = resolve_link(url, "search", "", "", index=data.read_index(INDEX), http=client)

    assert c.url == (
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/births/"
        "US_births_2000-2014_SSA.csv"
    )
    assert c.license == "CC-BY-4.0"


def test_resolve_named_github_folder_lists_its_data_files(http) -> None:
    client, seen = http

    (c,) = resolve_link(
        "https://github.com/fivethirtyeight/data/tree/master/births",
        "search",
        "births",
        "",
        index=data.read_index(INDEX),
        http=client,
    )

    assert c.url.endswith("/births/US_births_2000-2014_SSA.csv")
    assert c.bytes == 96748
    assert seen == ["https://api.github.com/repos/fivethirtyeight/data/contents/births?ref=master"]


@pytest.mark.parametrize(
    "link",
    [
        "https://higgi13425.github.io/medicaldata/reference/licorice_gargle.html",  # pkgdown
        f"{RD}/doc/medicaldata/licorice_gargle.html",
    ],
)
def test_resolve_package_docs_to_rdatasets_csv(link: str, http) -> None:
    client, seen = http

    (c,) = resolve_link(link, "paper", "", "", index=data.read_index(INDEX), http=client)

    assert (c.url, c.license, c.source) == (LICORICE, "MIT", "paper")
    assert "235 rows" in c.description
    assert seen == []


def test_other_links_are_kept_for_the_rule_filter(http) -> None:
    client, _ = http

    (c,) = resolve_link(
        "https://zenodo.org/records/1", "search", "t", "", index=data.read_index(INDEX), http=client
    )

    assert c.url == "https://zenodo.org/records/1"


def test_d1_data_found_scored_downloaded_and_profiled(run_find, fake_tavily) -> None:
    lacks_outcome = {"licorice_gargle": "treat", "sore_throat": None}
    found, state, fake = run_find([coverage(FULL, lacks_outcome)])

    assert found is not None and found.url == LICORICE
    assert (found.accepted, found.rank, found.score) == (True, 1, 1.0)
    assert found.columns == ["rownames", "treat", "pacu30min_throatPain", "preOp_smoking"]
    assert found.coverage == FULL
    assert (found.source, found.study_id) == ("paper", "s1")  # the trial's own data link
    (d,) = state.downloads
    assert (d.url, d.license) == (LICORICE, "MIT")
    assert state.profile is not None and state.profile.rows == 2
    copy = next(c for c in state.datasets if c.url == LICORICE_COPY)
    assert copy.rejected_reason == "lacks sore_throat" and copy.rank == 2
    # the PubMed hit from Tavily is kept as a rejected candidate
    assert any("allowlist" in (c.rejected_reason or "") for c in state.datasets)
    assert fake_tavily.requests[0]["include_domains_mode"] == "restrict"
    assert "licorice gargle sore throat" in fake_tavily.requests[0]["query"]
    # the model sees column names, never rows
    (request,) = fake.requests
    user = request["messages"][1]["content"]
    assert "pacu30min_throatPain" in user and "2,1,0,0" not in user
    assert "mtcars" not in user  # no keyword match → never scored


def test_failed_download_moves_to_the_next_rank(run_find, served, monkeypatch) -> None:
    calls: list[str] = []
    real = data.download

    def flaky(url: str, *args, **kwargs):
        if url == data.RDATASETS_INDEX:
            return real(url, *args, **kwargs)
        calls.append(url)
        if url == LICORICE:
            raise dl.DownloadRejected("size exceeds cap")
        return real(url, *args, **kwargs)

    monkeypatch.setattr(data, "download", flaky)

    found, state, _ = run_find([coverage(FULL, FULL)])

    assert found is not None and (found.url, found.rank) == (LICORICE_COPY, 2)
    first = next(c for c in state.datasets if c.url == LICORICE)
    assert first.rejected_reason == "download failed: size exceeds cap"
    assert calls == [LICORICE, LICORICE_COPY]


def test_multi_table_zip_is_not_usable_and_next_is_tried(run_find, monkeypatch, tmp_path) -> None:
    real = data.download

    def zipped(url: str, *args, **kwargs):
        d = real(url, *args, **kwargs)
        if url == LICORICE:
            path = tmp_path / "two.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("a.csv", "x\n1\n")
                z.writestr("b.csv", "y\n1\n")
            d.path = str(path)
        return d

    monkeypatch.setattr(data, "download", zipped)

    found, state, _ = run_find([coverage(FULL, FULL)])

    assert found is not None and found.url == LICORICE_COPY
    first = next(c for c in state.datasets if c.url == LICORICE)
    assert first.rejected_reason is not None and "2 tables" in first.rejected_reason


def test_at_most_three_downloads(run_find, served, monkeypatch) -> None:
    for i in range(4):
        served[f"{RD}/csv/medicaldata/licorice_{i}.csv"] = TRIAL_CSV
    served[data.RDATASETS_INDEX] = index_csv(
        *[("medicaldata", f"licorice_{i}", "Licorice gargle trial") for i in range(4)]
    )
    calls: list[str] = []
    real = data.download

    def failing(url: str, *args, **kwargs):
        if url == data.RDATASETS_INDEX:
            return real(url, *args, **kwargs)
        calls.append(url)
        raise dl.DownloadRejected("nope")

    monkeypatch.setattr(data, "download", failing)

    found, _, _ = run_find([coverage(*[FULL] * 4)])

    assert found is None
    assert len(calls) == 3


def test_missing_variable_triggers_one_re_search_naming_it(run_find, fake_tavily) -> None:
    lacks_outcome = {"licorice_gargle": "treat", "sore_throat": None}

    found, state, fake = run_find([coverage(lacks_outcome, lacks_outcome)])

    assert found is None
    assert len(fake_tavily.requests) == 2
    assert fake_tavily.requests[1]["query"] == "sore throat data"
    assert len(fake.requests) == 1  # the re-search found nothing new to score
    assert all(not c.accepted for c in state.datasets)


def test_only_the_top_ten_are_peeked_and_scored(run_find, served, http) -> None:
    rows = [("medicaldata", f"licorice_{i:02d}", "Licorice gargle trial") for i in range(12)]
    served[data.RDATASETS_INDEX] = index_csv(*rows)
    for _, item, _ in rows:
        served[f"{RD}/csv/medicaldata/{item}.csv"] = TRIAL_CSV
    _, seen = http

    found, state, _ = run_find([coverage(*[FULL] * 10)])

    assert found is not None
    peeked = [u for u in seen if "/csv/medicaldata/" in u]
    assert len(peeked) == 11  # 10 peeks + 1 download
    outside = [c for c in state.datasets if "outside the top 10" in (c.rejected_reason or "")]
    assert len(outside) == 2


def test_outcome_keyword_ranks_first_for_the_top_ten(run_find, served, http) -> None:
    rows = [("pkg", f"licorice_{i:02d}", "Licorice gargle survey") for i in range(11)]
    rows.append(("pkg", "zz_throat", "Sore throat after surgery"))
    served[data.RDATASETS_INDEX] = index_csv(*rows)
    for _, item, _ in rows:
        served[f"{RD}/csv/pkg/{item}.csv"] = TRIAL_CSV
    _, seen = http

    run_find([coverage(*[FULL] * 10)])

    assert f"{RD}/csv/pkg/zz_throat.csv" in seen


def test_keywords_match_at_word_starts_only() -> None:
    c = candidate(f"{RD}/csv/x/a.csv", "search", "Cutthroat trout; US_births", "")

    assert keyword_hits(c, ["throat", "birth"]) == 1
