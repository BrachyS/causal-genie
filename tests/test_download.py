import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from causal_genie.tools import download as dl
from causal_genie.tools.download import DownloadRejected, allowlisted, download, peek_columns

CSV_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/licorice_gargle.csv"
BODY = b"treat,pain\n1,0\n0,3\n"


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every host to a public address unless a test says otherwise."""
    monkeypatch.setattr(dl, "resolve_ips", lambda host: ["185.199.108.153"])


@pytest.fixture
def make_http() -> Callable[..., tuple[httpx.Client, list[httpx.Request]]]:
    """Build a mock HTTP client from a handler; also returns the requests it saw."""

    def make(handler: Callable[[httpx.Request], httpx.Response]):
        seen: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        return httpx.Client(transport=httpx.MockTransport(record)), seen

    return make


def test_allowlisted_csv_is_saved_with_provenance(make_http, tmp_path: Path) -> None:
    http, _ = make_http(lambda r: httpx.Response(200, content=BODY))

    d = download(CSV_URL, http, tmp_path, doi=None, license="MIT")

    assert d.sha256 == hashlib.sha256(BODY).hexdigest()
    assert d.bytes == len(BODY)
    assert Path(d.path).read_bytes() == BODY
    assert Path(d.path).suffix == ".csv"
    assert (d.url, d.license) == (CSV_URL, "MIT")


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("https://example.com/data.csv", "allowlist"),
        ("https://raw.githubusercontent.com/someone/repo/main/data.csv", "allowlist"),
        ("http://vincentarelbundock.github.io/Rdatasets/csv/x.csv", "https"),
        ("https://vincentarelbundock.github.io/Rdatasets/x.rda", "format"),
        ("https://zenodo.org/records/1/files/run.py", "format"),
    ],
)
def test_rejected_urls_are_never_fetched(make_http, tmp_path: Path, url: str, reason: str) -> None:
    http, seen = make_http(lambda r: httpx.Response(200, content=BODY))

    with pytest.raises(DownloadRejected, match=reason):
        download(url, http, tmp_path, doi=None, license=None)
    assert seen == []


def test_named_github_repo_is_allowed(make_http, tmp_path: Path) -> None:
    http, _ = make_http(lambda r: httpx.Response(200, content=BODY))
    url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births.csv"

    assert download(url, http, tmp_path, doi=None, license="CC-BY-4.0").bytes == len(BODY)


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1"])
def test_private_network_address_is_rejected(
    make_http, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ip: str
) -> None:
    monkeypatch.setattr(dl, "resolve_ips", lambda host: [ip])
    http, seen = make_http(lambda r: httpx.Response(200, content=BODY))

    with pytest.raises(DownloadRejected, match="private"):
        download(CSV_URL, http, tmp_path, doi=None, license=None)
    assert seen == []


def test_redirect_off_the_allowlist_is_rejected(make_http, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.example/data.csv"})

    http, seen = make_http(handler)

    with pytest.raises(DownloadRejected, match="allowlist"):
        download(CSV_URL, http, tmp_path, doi=None, license=None)
    assert len(seen) == 1


def test_redirect_within_allowlist_is_followed(make_http, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("old.csv"):
            return httpx.Response(301, headers={"location": "/Rdatasets/csv/new.csv"})
        return httpx.Response(200, content=BODY)

    http, seen = make_http(handler)
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/old.csv"

    assert download(url, http, tmp_path, doi=None, license=None).bytes == len(BODY)
    assert len(seen) == 2


def test_declared_size_over_cap_is_rejected(make_http, tmp_path: Path) -> None:
    http, _ = make_http(lambda r: httpx.Response(200, headers={"content-length": "999"}))

    with pytest.raises(DownloadRejected, match="size"):
        download(CSV_URL, http, tmp_path, doi=None, license=None, max_bytes=100)


def test_streamed_size_over_cap_is_rejected(make_http, tmp_path: Path) -> None:
    # an iterator body is sent chunked, with no content-length to check up front
    http, _ = make_http(lambda r: httpx.Response(200, content=iter([b"x" * 60, b"x" * 60])))

    with pytest.raises(DownloadRejected, match="size"):
        download(CSV_URL, http, tmp_path, doi=None, license=None, max_bytes=100)
    assert list(tmp_path.iterdir()) == []


def test_http_error_is_raised(make_http, tmp_path: Path) -> None:
    http, _ = make_http(lambda r: httpx.Response(404))

    with pytest.raises(httpx.HTTPStatusError):
        download(CSV_URL, http, tmp_path, doi=None, license=None)


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        (CSV_URL, True),
        ("https://github.com/fivethirtyeight/data/tree/master/births", True),
        ("https://github.com/someone/repo/blob/main/x.csv", False),
        ("http://zenodo.org/records/1", False),
        ("https://www.vivli.org/study/1", False),
    ],
)
def test_allowlisted_checks_scheme_host_and_path_without_dns(
    monkeypatch: pytest.MonkeyPatch, url: str, ok: bool
) -> None:
    monkeypatch.setattr(dl, "resolve_ips", lambda host: pytest.fail("no DNS lookup"))

    assert allowlisted(url) is ok


def test_peek_reads_only_the_header_line(make_http) -> None:
    body = iter([b'"rownames","treat","pacu30min_', b'throatPain"\n1,0,3\n', b"2,1,0\n" * 1000])
    http, _ = make_http(lambda r: httpx.Response(200, content=body))

    assert peek_columns(CSV_URL, http) == ["rownames", "treat", "pacu30min_throatPain"]


def test_peek_splits_tsv_on_tabs(make_http) -> None:
    http, _ = make_http(lambda r: httpx.Response(200, content=b"a\tb\n1\t2\n"))
    url = "https://zenodo.org/records/1/files/t.tsv"

    assert peek_columns(url, http) == ["a", "b"]


def test_peek_of_other_formats_or_off_allowlist(make_http) -> None:
    http, seen = make_http(lambda r: httpx.Response(200, content=BODY))

    assert peek_columns("https://zenodo.org/records/1/files/t.xlsx", http) == []
    with pytest.raises(DownloadRejected, match="allowlist"):
        peek_columns("https://example.com/t.csv", http)
    assert seen == []


def test_peek_without_header_line_within_cap_is_rejected(make_http) -> None:
    http, _ = make_http(lambda r: httpx.Response(200, content=b"x" * 500))

    with pytest.raises(DownloadRejected, match="header"):
        peek_columns(CSV_URL, http, max_bytes=100)


# --- Fetcher: records every data-step request so replay needs no network (T11.4) ---

LISTING_URL = "https://api.github.com/repos/fivethirtyeight/data/contents/births"
LISTING = [
    {"name": "README.md", "type": "file", "size": 826, "download_url": "https://x/README.md"},
    {"name": "sub", "type": "dir", "size": 0, "download_url": None},
]


@pytest.fixture
def no_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "resolve_ips", lambda host: pytest.fail(f"DNS lookup of {host}"))


def test_fetcher_records_downloads_peeks_and_listings(make_http, tmp_path: Path) -> None:
    def handle(r: httpx.Request) -> httpx.Response:
        if r.url.host == "api.github.com":
            assert r.url.params["ref"] == "master"
            return httpx.Response(200, json=LISTING)
        return httpx.Response(200, content=BODY)

    http, _ = make_http(handle)
    fetcher = dl.Fetcher(http)

    d = fetcher.download(CSV_URL, tmp_path, doi=None, license="MIT")
    assert fetcher.peek(CSV_URL) == ["treat", "pain"]
    files = fetcher.listing("fivethirtyeight", "data", "master", "births")

    assert [f.name for f in files] == ["README.md", "sub"] and files[1].download_url is None
    kinds = [(c.kind, c.url) for c in fetcher.calls]
    assert kinds == [
        ("download", CSV_URL),
        ("peek", CSV_URL),
        ("listing", f"{LISTING_URL}?ref=master"),
    ]
    assert fetcher.calls[0].download == d and fetcher.calls[1].columns == ["treat", "pain"]
    assert all(c.error is None and c.latency_s >= 0 for c in fetcher.calls)


def test_fetcher_replay_makes_no_request_or_dns_lookup(make_http, tmp_path: Path, no_dns) -> None:
    def handle(r: httpx.Request) -> httpx.Response:
        if r.url.host == "api.github.com":
            return httpx.Response(200, json=LISTING)
        return httpx.Response(200, content=BODY)

    http, _ = make_http(handle)
    live = dl.Fetcher(http)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(dl, "resolve_ips", lambda host: ["185.199.108.153"])
        d = live.download(CSV_URL, tmp_path, doi=None, license="MIT")
        columns = live.peek(CSV_URL)
        files = live.listing("fivethirtyeight", "data", "master", "births")

    replay = dl.Fetcher.from_replay(live.calls)
    assert replay.download(CSV_URL, tmp_path, doi=None, license="MIT") == d
    assert replay.peek(CSV_URL) == columns
    assert replay.listing("fivethirtyeight", "data", "master", "births") == files
    assert [(c.kind, c.url) for c in replay.calls] == [(c.kind, c.url) for c in live.calls]


def test_fetcher_replays_recorded_failures_as_the_same_errors(make_http, tmp_path: Path) -> None:
    http, _ = make_http(lambda r: httpx.Response(404))
    live = dl.Fetcher(http)
    with pytest.raises(httpx.HTTPStatusError) as peek_error:
        live.peek(CSV_URL)
    with pytest.raises(DownloadRejected) as rejected:
        live.download("https://example.com/t.csv", tmp_path, doi=None, license=None)

    replay = dl.Fetcher.from_replay(live.calls)
    with pytest.raises(httpx.HTTPError) as replayed_peek:
        replay.peek(CSV_URL)
    with pytest.raises(DownloadRejected) as replayed_rejection:
        replay.download("https://example.com/t.csv", tmp_path, doi=None, license=None)
    assert str(replayed_peek.value) == str(peek_error.value)  # same rejection reason on the run
    assert str(replayed_rejection.value) == str(rejected.value)


def test_fetcher_replay_miss_raises(tmp_path: Path) -> None:
    replay = dl.Fetcher.from_replay([])

    with pytest.raises(KeyError, match="no recorded peek"):
        replay.peek(CSV_URL)


def test_fetcher_replay_checks_the_cached_file(make_http, tmp_path: Path) -> None:
    http, _ = make_http(lambda r: httpx.Response(200, content=BODY))
    live = dl.Fetcher(http)
    d = live.download(CSV_URL, tmp_path, doi=None, license=None)
    replay = dl.Fetcher.from_replay(live.calls)

    Path(d.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        replay.download(CSV_URL, tmp_path, doi=None, license=None)
    Path(d.path).unlink()
    with pytest.raises(FileNotFoundError, match="re-record"):
        replay.download(CSV_URL, tmp_path, doi=None, license=None)


# --- Repository files (T14.1, T14.2): formats from file names, storage redirects, records ---

DATAVERSE_FILE = "https://dataverse.harvard.edu/api/access/datafile/4461133"
FIGSHARE_FILE = "https://ndownloader.figshare.com/files/69927615"
FIGSHARE_STORAGE = (
    "https://44c1cec316ec56bca76dfd65d2882b20.r2.cloudflarestorage.com/figshare-com-storage/"
    "69927615/growth_form_raw.csv?X-Amz-Signature=abc"
)
DATAVERSE_STORAGE = "https://dvn-cloud-iqss.s3.amazonaws.com/10.7910/DVN/SAWC6M/1784b4f3baf?sig=1"


def test_file_url_without_extension_uses_the_format_from_its_name(
    make_http, tmp_path: Path
) -> None:
    http, _ = make_http(lambda r: httpx.Response(200, content=b"a\tb\n1\t2\n"))

    d = download(
        DATAVERSE_FILE, http, tmp_path, doi="10.7910/DVN/SAWC6M", license=None, suffix=".tsv"
    )

    assert Path(d.path).suffix == ".tsv"
    assert d.doi == "10.7910/DVN/SAWC6M"


@pytest.mark.parametrize("suffix", [".sav", None])
def test_file_url_with_a_non_data_or_unknown_format_is_rejected(
    make_http, tmp_path: Path, suffix: str | None
) -> None:
    http, seen = make_http(lambda r: httpx.Response(200, content=BODY))

    with pytest.raises(DownloadRejected, match="format"):
        download(DATAVERSE_FILE, http, tmp_path, doi=None, license=None, suffix=suffix)
    assert seen == []


@pytest.mark.parametrize(
    ("start", "storage"),
    [(FIGSHARE_FILE, FIGSHARE_STORAGE), (DATAVERSE_FILE, DATAVERSE_STORAGE)],
)
def test_redirect_to_the_repository_storage_bucket_is_followed(
    make_http, tmp_path: Path, start: str, storage: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == start:
            return httpx.Response(302, headers={"location": storage})
        return httpx.Response(200, content=BODY)

    http, seen = make_http(handler)

    assert download(start, http, tmp_path, doi=None, license=None, suffix=".csv").bytes == len(BODY)
    assert len(seen) == 2


@pytest.mark.parametrize(
    "url",
    [
        "https://44c1cec316ec56bca76dfd65d2882b20.r2.cloudflarestorage.com/other-bucket/x.csv",
        "https://other.r2.cloudflarestorage.com/figshare-com-storage/x.csv",
        "https://another-bucket.s3.amazonaws.com/x.csv",
    ],
)
def test_other_storage_buckets_are_not_allowlisted(url: str) -> None:
    assert allowlisted(url) is False


def test_peek_follows_allowlisted_redirects_and_uses_the_given_format(make_http) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == DATAVERSE_FILE:
            return httpx.Response(303, headers={"location": DATAVERSE_STORAGE})
        return httpx.Response(200, content=b"a\tb\n1\t2\n")

    http, seen = make_http(handler)

    assert peek_columns(DATAVERSE_FILE, http, suffix=".tsv") == ["a", "b"]
    assert len(seen) == 2


def test_peek_rejects_a_redirect_off_the_allowlist(make_http) -> None:
    http, seen = make_http(
        lambda r: httpx.Response(302, headers={"location": "https://evil.example/t.csv"})
    )

    with pytest.raises(DownloadRejected, match="allowlist"):
        peek_columns(CSV_URL, http)
    assert len(seen) == 1


ZENODO_API = "https://zenodo.org/api/records/7826073"
ZENODO_RECORD = {
    "id": 7826073,
    "doi": "10.5281/zenodo.7826073",
    "metadata": {"title": "Cloud trails", "license": {"id": "cc-by-4.0"}},
    "files": [{"key": "fig8_data.csv", "size": 626}],
}


def test_fetcher_records_and_replays_a_repository_record(make_http, no_dns) -> None:
    http, seen = make_http(lambda r: httpx.Response(200, json=ZENODO_RECORD))
    live = dl.Fetcher(http)

    files = live.record("https://zenodo.org/records/7826073")

    assert [f.name for f in files or []] == ["fig8_data.csv"]
    assert [str(r.url) for r in seen] == [ZENODO_API]
    assert [(c.kind, c.url) for c in live.calls] == [("record", ZENODO_API)]
    replay = dl.Fetcher.from_replay(live.calls)
    assert replay.record("https://doi.org/10.5281/zenodo.7826073") == files


def test_fetcher_record_of_a_non_record_link_makes_no_request(make_http) -> None:
    http, seen = make_http(lambda r: httpx.Response(200, json={}))

    assert dl.Fetcher(http).record("https://osf.io/xdpz5/") is None
    assert seen == []


def test_fetcher_records_a_failed_record_lookup_and_replays_it(make_http) -> None:
    http, _ = make_http(lambda r: httpx.Response(410))
    live = dl.Fetcher(http)

    with pytest.raises(httpx.HTTPStatusError):
        live.record("https://zenodo.org/records/7826073")

    assert live.calls[0].error_type == "http"
    with pytest.raises(httpx.HTTPError):
        dl.Fetcher.from_replay(live.calls).record("https://zenodo.org/records/7826073")


def test_peek_drops_a_byte_order_mark(make_http) -> None:
    http, _ = make_http(lambda r: httpx.Response(200, content="\ufeffsite,year\n1,2\n".encode()))

    assert peek_columns(CSV_URL, http) == ["site", "year"]
