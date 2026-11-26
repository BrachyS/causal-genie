import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import httpx
import pandas as pd
import pytest
from conftest import FIXTURES
from tavily import TavilyClient

from causal_genie.agent.state import DagEdge, DatasetCandidate, RunState
from causal_genie.agent.steps import data
from causal_genie.agent.steps.data import (
    candidate,
    find_data,
    index_candidate,
    keyword_hits,
    keyword_weights,
    keywords,
    read_index,
    resolve_link,
    rule_filter,
    shortlist,
    use_table,
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
            sample_state,
            llm,
            Searcher(cast(TavilyClient, fake_tavily)),
            dl.Fetcher(client),
            tmp_path,
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

    (c,) = resolve_link(
        url, "search", "", "", index=data.read_index(INDEX), fetcher=dl.Fetcher(client)
    )

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
        fetcher=dl.Fetcher(client),
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

    (c,) = resolve_link(
        link, "paper", "", "", index=data.read_index(INDEX), fetcher=dl.Fetcher(client)
    )

    assert (c.url, c.license, c.source) == (LICORICE, "MIT", "paper")
    assert "235 rows" in c.description
    assert seen == []


def test_other_links_are_kept_for_the_rule_filter(http) -> None:
    client, _ = http

    (c,) = resolve_link(
        "https://osf.io/xdpz5/",
        "search",
        "t",
        "",
        index=data.read_index(INDEX),
        fetcher=dl.Fetcher(client),
    )

    assert c.url == "https://osf.io/xdpz5/"


# --- Repository records → files (T14.1, T14.2) ---

REPOS = FIXTURES / "repositories"
ZENODO_API = "https://zenodo.org/api/records/7826073"
DATAVERSE_API = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/SAWC6M"
)
DATAVERSE_FILE = "https://dataverse.harvard.edu/api/access/datafile/4461133"


def test_zenodo_record_page_becomes_one_candidate_per_data_file(http, served) -> None:
    client, seen = http
    served[ZENODO_API] = (REPOS / "zenodo_record.json").read_bytes()

    cands = resolve_link(
        "https://zenodo.org/records/7826073",
        "search",
        "Cloud trail data",
        "snippet",
        index=data.read_index(INDEX),
        fetcher=dl.Fetcher(client),
    )

    assert [c.title for c in cands] == [
        "Cloud trail data: fig8_data.csv",
        "Cloud trail data: fig1_simplified_data.csv",
    ]  # the .nc file is not a table we read
    assert cands[0].url == "https://zenodo.org/records/7826073/files/fig8_data.csv?download=1"
    assert (cands[0].license, cands[0].doi, cands[0].bytes, cands[0].format) == (
        "cc-by-4.0",
        "10.5281/zenodo.7826073",
        626,
        ".csv",
    )
    assert seen == [ZENODO_API]


def test_dataverse_doi_from_a_paper_gives_a_tsv_candidate_that_passes_the_rules(
    http, served
) -> None:
    client, _ = http
    served[DATAVERSE_API] = (REPOS / "dataverse_dataset.json").read_bytes()

    (c,) = resolve_link(
        "https://doi.org/10.7910/DVN/SAWC6M",
        "paper",
        "data link from Smith 2019",
        "",
        index=data.read_index(INDEX),
        fetcher=dl.Fetcher(client),
    )
    rule_filter(c, ["parasitoid"])

    assert (c.url, c.format, c.study_id) == (DATAVERSE_FILE, ".tsv", None)
    assert "experiment1_parasitoids.tab" in c.title
    assert (c.rejected_reason, c.access, c.license) == (None, "open", "CC0 1.0")


def test_record_title_is_used_when_the_link_has_none(http, served) -> None:
    client, _ = http
    served[ZENODO_API] = (REPOS / "zenodo_record.json").read_bytes()

    cands = resolve_link(
        "https://doi.org/10.5281/zenodo.7826073",
        "paper",
        "",
        "",
        index=data.read_index(INDEX),
        fetcher=dl.Fetcher(client),
    )

    assert cands[0].title == ("Sensible heat fluxes control cloud trail strength: fig8_data.csv")


def test_failed_record_lookup_keeps_the_page_for_the_rule_filter(http) -> None:
    client, _ = http  # the API URL is not served: 404
    fetcher = dl.Fetcher(client)

    (c,) = resolve_link(
        "https://zenodo.org/records/7826073",
        "search",
        "t",
        "",
        index=data.read_index(INDEX),
        fetcher=fetcher,
    )

    assert c.url == "https://zenodo.org/records/7826073"
    assert fetcher.calls[0].error_type == "http"


def test_file_without_extension_in_its_url_passes_the_rule_filter_by_its_format() -> None:
    c = make("https://ndownloader.figshare.com/files/69927615", format=".csv")

    rule_filter(c, KW)

    assert c.rejected_reason is None


def test_pick_downloads_with_the_candidate_format_and_doi(
    http, served, sample_state: RunState, tmp_path: Path
) -> None:
    client, _ = http
    served[DATAVERSE_FILE] = b"treat\tpacu30min_throatPain\n1\t0\n0\t3\n"
    c = make(DATAVERSE_FILE, format=".tsv", doi="10.7910/DVN/SAWC6M", rank=1, score=1.0)
    sample_state.datasets = [c]

    assert data.pick(sample_state, dl.Fetcher(client), tmp_path, max_downloads=3) is c

    (d,) = sample_state.downloads
    assert (Path(d.path).suffix, d.doi) == (".tsv", "10.7910/DVN/SAWC6M")


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
    real = dl.download

    def flaky(url: str, *args, **kwargs):
        if url == data.RDATASETS_INDEX:
            return real(url, *args, **kwargs)
        calls.append(url)
        if url == LICORICE:
            raise dl.DownloadRejected("size exceeds cap")
        return real(url, *args, **kwargs)

    monkeypatch.setattr(dl, "download", flaky)

    found, state, _ = run_find([coverage(FULL, FULL)])

    assert found is not None and (found.url, found.rank) == (LICORICE_COPY, 2)
    first = next(c for c in state.datasets if c.url == LICORICE)
    assert first.rejected_reason == "download failed: size exceeds cap"
    assert calls == [LICORICE, LICORICE_COPY]


def test_multi_table_zip_is_not_usable_and_next_is_tried(run_find, monkeypatch, tmp_path) -> None:
    real = dl.download

    def zipped(url: str, *args, **kwargs):
        d = real(url, *args, **kwargs)
        if url == LICORICE:
            path = tmp_path / "two.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("a.csv", "x\n1\n")
                z.writestr("b.csv", "y\n1\n")
            d.path = str(path)
        return d

    monkeypatch.setattr(dl, "download", zipped)

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
    real = dl.download

    def failing(url: str, *args, **kwargs):
        if url == data.RDATASETS_INDEX:
            return real(url, *args, **kwargs)
        calls.append(url)
        raise dl.DownloadRejected("nope")

    monkeypatch.setattr(dl, "download", failing)

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


def test_user_table_is_profiled_as_the_accepted_candidate(sample_state: RunState) -> None:
    path = FIXTURES / "d3_onboarding_ab.csv"

    accepted = use_table(sample_state, path)

    assert accepted.accepted and accepted.source == "user" and accepted.rank == 1
    assert sample_state.datasets == [accepted]
    assert sample_state.profile is not None and sample_state.profile.rows == 6000
    (d,) = sample_state.downloads
    assert d.path == str(path) and d.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "new_flow" in accepted.columns


def sized_index(*rows: tuple[str, str, str, int]) -> pd.DataFrame:
    """A parsed Rdatasets index with (package, item, title, row count) rows."""
    lines = [
        ROW.format(pkg=p, item=i, title=t, rd=RD).replace(",235,", f",{n},", 1)
        for p, i, t, n in rows
    ]
    return read_index("\n".join([INDEX_HEADER, *lines]).encode())


# Live D0 (run 20261114-195218-618018): the outcome was named `birth_count`, so "count"
# (COUNT package, Country, County, Counts of ...) tied with "birth" and the US births tables
# fell outside the top 10 alphabetically.
D0_INDEX = sized_index(
    *[("COUNT", f"c{i}", f"Counts of things {i}", 100) for i in range(6)],
    ("admiral", "country_code_lookup", "Country Code Lookup", 250),
    ("AER", "OrangeCounty", "Orange County Employment", 52),
    ("openintro", "present", "Birth counts", 63),
    ("fivethirtyeight", "US_births_1994_2003", "Some People Are Too Superstitious", 3652),
    ("mosaicData", "Births", "US Births", 7305),
    *[("filler", f"f{i}", f"Unrelated table {i}", 10) for i in range(20)],
)


def as_candidates(index: pd.DataFrame) -> list[DatasetCandidate]:
    return [index_candidate(row, "repository") for _, row in index.iterrows()]


def test_rare_words_weigh_more_and_unknown_words_most() -> None:
    w = keyword_weights(D0_INDEX, ["birth", "count", "zzz"])

    assert w["zzz"] > w["birth"] > w["count"]
    assert w["zzz"] == pytest.approx(math.log(1 + len(D0_INDEX)))


def test_d0_shortlist_puts_birth_tables_before_count_and_country_tables() -> None:
    kws, lead = keywords(["full_moon", "birth_count"]), keywords(["birth_count"])

    top = shortlist(as_candidates(D0_INDEX), kws, lead, D0_INDEX)

    titles = [c.title.split(":")[0] for c in top[:3]]
    assert titles == ["openintro", "mosaicData", "fivethirtyeight"]  # birth AND count first


def test_shortlist_ties_go_to_the_larger_table() -> None:
    kws = lead = keywords(["births"])

    top = shortlist(as_candidates(D0_INDEX), kws, lead, D0_INDEX)

    assert [c.url.split("/")[-1] for c in top[:3]] == [
        "Births.csv",
        "US_births_1994_2003.csv",
        "present.csv",
    ]


# Live D0 run 20261114-211437-12b84e: the small model marked `full_moon` uncovered in every
# table, though six have a `date` column. Metadata only (titles, URLs, headers), no data rows.
D0_RUN4 = json.loads((FIXTURES / "d0_run4_coverage.json").read_text())


def d0_run4_scored(make_llm) -> list[DatasetCandidate]:
    """Score the run's candidates with the model's recorded coverage replayed."""
    state = RunState(
        run_id="t",
        question="Does the full moon cause more births?",
        query=D0_RUN4["query"],
        edges=D0_RUN4["edges"],
        exogenous=D0_RUN4["exogenous"],
    )
    cands = [DatasetCandidate.model_validate(c["candidate"]) for c in D0_RUN4["candidates"]]
    llm, _ = make_llm([coverage(*[c["model_coverage"] for c in D0_RUN4["candidates"]])])
    data.score(cands, state, llm)
    return cands


def test_d0_run4_daily_us_tables_cover_full_moon_from_their_date(make_llm) -> None:
    cands = d0_run4_scored(make_llm)

    daily, australian = cands[:6], cands[6]
    for c in daily:
        assert c.rejected_reason is None, c.title
        assert c.coverage["full_moon"] == "date"
        assert c.coverage["day_of_week"] in ("day_of_week", "wday")  # the model's column is kept
    assert australian.coverage["full_moon"] is None  # Month only: no calendar day
    assert australian.rejected_reason == "lacks full_moon"


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["date", "births"], "date"),
        (["Birth_Date", "births"], "Birth_Date"),
        (["Year", "Month", "Day", "births"], "Year, Month, Day"),
        (["year", "month", "date_of_month", "births"], "year, month, date_of_month"),
        (["year", "month", "day_of_month", "births"], "year, month, day_of_month"),
        (["year", "month", "mday", "births"], "year, month, mday"),
        (["date_of_month", "births"], None),  # a day number is not a date
        (["year", "month", "births"], None),
        ([], None),  # header unknown
    ],
)
def test_calendar_columns(columns: list[str], expected: str | None) -> None:
    assert data.calendar_columns(columns) == expected


def test_harness_covers_only_calendar_nodes_the_model_left_null(make_llm, sample_state) -> None:
    sample_state.query.exposure.name = "full_moon"  # type: ignore[union-attr]
    sample_state.query.outcome.name = "births"  # type: ignore[union-attr]
    sample_state.edges = [
        DagEdge(source=s, target="births", claim_ids=[])
        for s in ("full_moon", "holiday", "season", "weather")
    ]
    c = candidate(BIRTHS, "repository", "US births", "")
    c.columns = ["date", "births", "month"]
    llm, _ = make_llm([coverage({"births": "births", "season": "month"})])

    data.score([c], sample_state, llm)

    assert c.coverage == {
        "full_moon": "date",
        "births": "births",
        "holiday": "date",
        "season": "month",  # the model's column is kept
        "weather": None,  # not a calendar rule
    }
    assert c.score == 4 / 5 and c.rejected_reason is None


def test_dryad_page_without_keys_is_kept_with_the_reason(http) -> None:
    client, seen = http
    page = "https://datadryad.org/dataset/doi:10.5061/dryad.4b8gthtrz"

    (c,) = resolve_link(
        page,
        "search",
        "licorice gargle",
        "",
        index=data.read_index(INDEX),
        fetcher=dl.Fetcher(client),
    )
    rule_filter(c, KW)

    assert c.url == page
    assert "DRYAD_CLIENT_ID" in (c.rejected_reason or "")
    assert seen == []


def test_rule_filter_keeps_a_reason_given_earlier() -> None:
    c = make(LICORICE, rejected_reason="record lookup failed: 410 Gone")

    rule_filter(c, KW)

    assert (c.rejected_reason, c.access) == ("record lookup failed: 410 Gone", "open")


# --- One candidate per sheet or ZIP member (T14.8) ---

WORKBOOK_URL = "https://zenodo.org/records/9/files/licorice_trial.xlsx?download=1"


def xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        for name, frame in sheets.items():
            frame.to_excel(w, sheet_name=name, index=False)
    return buf.getvalue()


TRIAL_SHEETS = {
    "1. Patients": pd.DataFrame({"id": [1, 2], "age": [40, 50]}),
    "1. Metadata": pd.DataFrame({"Notes": ["id is the patient"]}),
    "2. Outcomes": pd.DataFrame({"treat": [1, 0], "pacu30min_throatPain": [0, 3]}),
}


@pytest.fixture
def screen_workbook(make_llm, http, served, sample_state: RunState, tmp_path: Path):
    """Screen one licorice workbook candidate, with the model's coverage outputs given."""

    def run(body: bytes, *outcomes: str):
        served[WORKBOOK_URL] = body
        llm, _ = make_llm(list(outcomes))
        client, seen = http
        fetcher = dl.Fetcher(client)
        c = make(WORKBOOK_URL, format=".xlsx", license="cc-by-4.0")
        found = data.screen(
            [c], sample_state, llm, fetcher, KW, ["throat"], read_index(INDEX), tmp_path
        )
        return found, fetcher, seen

    return run


def test_workbook_becomes_one_scored_candidate_per_data_sheet(screen_workbook) -> None:
    found, _, seen = screen_workbook(
        xlsx_bytes(TRIAL_SHEETS),
        coverage({"licorice_gargle": None, "sore_throat": None}, FULL),
    )

    assert [(c.part, c.columns) for c in found] == [
        ("1. Patients", ["id", "age"]),
        ("2. Outcomes", ["treat", "pacu30min_throatPain"]),
    ]
    assert found[1].title == "licorice gargle data [sheet 2. Outcomes]"
    assert (found[0].rejected_reason, found[1].rejected_reason) == (
        "lacks licorice_gargle, sore_throat",
        None,
    )
    assert {c.license for c in found} == {"cc-by-4.0"}
    assert seen.count(WORKBOOK_URL) == 1


def test_chosen_sheet_is_downloaded_once_profiled_and_kept_on_the_download(
    screen_workbook, sample_state: RunState, tmp_path: Path
) -> None:
    found, fetcher, seen = screen_workbook(
        xlsx_bytes(TRIAL_SHEETS),
        coverage({"licorice_gargle": None, "sore_throat": None}, FULL),
    )
    sample_state.datasets = found
    data.rank(found)

    chosen = data.pick(sample_state, fetcher, tmp_path, max_downloads=3)

    assert chosen is not None and chosen.part == "2. Outcomes"
    assert sample_state.downloads[-1].part == "2. Outcomes"
    assert sample_state.profile is not None
    assert [c.name for c in sample_state.profile.columns] == ["treat", "pacu30min_throatPain"]
    assert seen.count(WORKBOOK_URL) == 1  # the screening download is reused


def test_single_sheet_workbook_is_scored_on_its_header(screen_workbook) -> None:
    sheet = {"Sheet1": TRIAL_SHEETS["2. Outcomes"]}
    (c,) = screen_workbook(xlsx_bytes(sheet), coverage(FULL))[0]

    assert (c.part, c.columns, c.rejected_reason) == (
        None,
        ["treat", "pacu30min_throatPain"],
        None,
    )


def test_unreadable_workbook_is_rejected_with_the_reason(screen_workbook) -> None:
    (c,) = screen_workbook(b"not a workbook")[0]

    assert (c.rejected_reason or "").startswith("tables could not be read")


def test_workbook_screening_replays_offline(
    screen_workbook, sample_state, make_llm, tmp_path
) -> None:
    out = coverage({"licorice_gargle": None, "sore_throat": None}, FULL)
    live, fetcher, _ = screen_workbook(xlsx_bytes(TRIAL_SHEETS), out)
    llm, _ = make_llm([out])

    replayed = data.screen(
        [make(WORKBOOK_URL, format=".xlsx", license="cc-by-4.0")],
        sample_state,
        llm,
        dl.Fetcher.from_replay(fetcher.calls),
        KW,
        ["throat"],
        read_index(INDEX),
        tmp_path,
    )

    assert [c.model_dump() for c in replayed] == [c.model_dump() for c in live]
