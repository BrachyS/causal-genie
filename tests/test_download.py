import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from causal_genie.tools import download as dl
from causal_genie.tools.download import DownloadRejected, download

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
