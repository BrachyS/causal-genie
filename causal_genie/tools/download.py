"""Fetch a dataset file safely: allowlisted hosts, data formats only, size-capped, public IPs only.

Downloads are data: nothing in them is ever executed, and they live under cache/ (never
committed). Each file is recorded with its checksum so estimates trace to exact bytes.
`Fetcher` records every request the data step makes, so a saved run replays offline.
"""

import csv
import hashlib
import ipaddress
import socket
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, TypeVar
from urllib.parse import urljoin, urlsplit

import httpx

from causal_genie.agent.state import Download, FetchCall, ListedFile, RepositoryFile
from causal_genie.log import log_step
from causal_genie.tools.repositories import (
    DRYAD_HOST,
    DryadAuth,
    fetch_record,
    is_dryad,
    record_api,
)

# host → allowed path prefixes (PRD §5.6). GitHub is limited to the named repos.
GITHUB_REPOS = ("/fivethirtyeight/data/", "/higgi13425/medicaldata/")
ALLOWLIST: dict[str, tuple[str, ...]] = {
    "zenodo.org": ("/",),
    "dataverse.harvard.edu": ("/",),
    "datadryad.org": ("/",),
    "figshare.com": ("/",),
    "ndownloader.figshare.com": ("/",),
    "osf.io": ("/",),
    "files.osf.io": ("/",),
    "vincentarelbundock.github.io": ("/Rdatasets/",),
    "clinicaltrials.gov": ("/",),
    "github.com": GITHUB_REPOS,
    "raw.githubusercontent.com": GITHUB_REPOS,
    # Storage that figshare and Harvard Dataverse downloads redirect to (signed links); only
    # these buckets, and only reached as a redirect hop from the repository's own URL.
    "44c1cec316ec56bca76dfd65d2882b20.r2.cloudflarestorage.com": ("/figshare-com-storage/",),
    "dvn-cloud-iqss.s3.amazonaws.com": ("/",),
    "dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com": ("/",),
}
FORMATS = {".csv", ".tsv", ".xlsx", ".parquet", ".zip"}
MAX_BYTES = 50 * 1024 * 1024
MAX_REDIRECTS = 5
PEEK_BYTES = 64 * 1024  # a header line longer than this is not a table we can use
GITHUB_API = "https://api.github.com/repos"
T = TypeVar("T")
FetchKind = Literal["download", "peek", "listing", "record"]


class DownloadRejected(ValueError):
    """The URL or response breaks a download rule; the caller may try the next candidate."""


def resolve_ips(host: str) -> list[str]:
    """Every IP `host` resolves to, so a public name pointing at a private network is caught."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, 443)]


def allowlisted(url: str) -> bool:
    """https on an allowlisted host and path; no network, so the rule filter can use it."""
    parts = urlsplit(url)
    prefixes = ALLOWLIST.get(parts.hostname or "", ())
    return parts.scheme == "https" and any(parts.path.startswith(p) for p in prefixes)


def check_url(url: str) -> None:
    """Raise `DownloadRejected` unless `url` is https, allowlisted and resolves to public IPs."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise DownloadRejected(f"https only: {url}")
    if not allowlisted(url):
        raise DownloadRejected(f"not on the allowlist: {url}")
    host = parts.hostname or ""
    private = [ip for ip in resolve_ips(host) if not ipaddress.ip_address(ip).is_global]
    if private:
        raise DownloadRejected(f"{host} resolves to private network address {private}")


def url_format(url: str, suffix: str | None) -> str:
    """The file format: `suffix` when given (a repository file URL may have no extension, so
    the format comes from the file's name), else the URL path's extension."""
    return suffix or PurePosixPath(urlsplit(url).path).suffix.lower()


@contextmanager
def open_checked(
    http: httpx.Client, url: str, dryad: DryadAuth | None = None
) -> Generator[httpx.Response]:
    """Stream `url`, following redirects; every hop is re-checked against the allowlist and
    the private-network rule before it is requested. With `dryad`, hops to datadryad.org
    carry its token; no other host (such as the storage a download redirects to) gets it."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_url(current)
        to_dryad = dryad is not None and urlsplit(current).hostname == DRYAD_HOST
        headers = dryad.headers(http) if dryad is not None and to_dryad else None
        with http.stream("GET", current, headers=headers, follow_redirects=False) as response:
            if response.is_redirect:
                current = urljoin(current, response.headers["location"])
                continue
            response.raise_for_status()
            yield response
            return
    raise DownloadRejected(f"more than {MAX_REDIRECTS} redirects: {url}")


def download(
    url: str,
    http: httpx.Client,
    cache_dir: Path,
    doi: str | None,
    license: str | None,
    max_bytes: int = MAX_BYTES,
    suffix: str | None = None,
    dryad: DryadAuth | None = None,
) -> Download:
    """Fetch `url` into `cache_dir` and return its provenance. `suffix` is the format from a
    repository file's name, for URLs without an extension; it must still be a data format."""
    suffix = url_format(url, suffix)
    with log_step("download", url=url) as rec:
        if suffix not in FORMATS:
            raise DownloadRejected(f"format {suffix or '(none)'} not in {sorted(FORMATS)}: {url}")
        with open_checked(http, url, dryad) as response:
            body = read_capped(response, max_bytes)
            final_url = str(response.url)

        sha256 = hashlib.sha256(body).hexdigest()
        path = cache_dir / f"{sha256}{suffix}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        rec.update(sha256=sha256, bytes=len(body), final_url=final_url)
        return Download(
            url=url,
            path=str(path),
            sha256=sha256,
            bytes=len(body),
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            doi=doi,
            license=license,
        )


def read_capped(response: httpx.Response, max_bytes: int) -> bytes:
    """Read the body, rejecting it as soon as it is known to exceed `max_bytes`."""
    declared = response.headers.get("content-length")
    if declared is not None and int(declared) > max_bytes:
        raise DownloadRejected(f"declared size {declared} bytes exceeds cap {max_bytes}")
    body = bytearray()
    for chunk in response.iter_bytes():
        body += chunk
        if len(body) > max_bytes:
            raise DownloadRejected(f"size exceeds cap {max_bytes} bytes")
    return bytes(body)


def peek_columns(
    url: str,
    http: httpx.Client,
    max_bytes: int = PEEK_BYTES,
    suffix: str | None = None,
    dryad: DryadAuth | None = None,
) -> list[str]:
    """Column names from a CSV or TSV header line, so candidates can be scored before download.

    Reads at most `max_bytes` and keeps only the first line; rows are never kept. Other formats
    return []. Redirects are followed with every hop checked, as in `download`; `suffix` is as
    there.
    """
    suffix = url_format(url, suffix)
    if suffix not in {".csv", ".tsv"}:
        return []
    with log_step("peek", url=url) as rec, open_checked(http, url, dryad) as response:
        head = bytearray()
        for chunk in response.iter_bytes():
            head += chunk
            if b"\n" in head or len(head) > max_bytes:
                break
        if b"\n" not in head[:max_bytes]:
            raise DownloadRejected(f"no header line within {max_bytes} bytes: {url}")
        line = head[: head.index(b"\n")].decode("utf-8-sig", errors="replace").rstrip("\r")
        columns = next(csv.reader([line], delimiter="\t" if suffix == ".tsv" else ","))
        rec["columns"] = columns
        return columns


class Fetcher:
    """Downloads, header reads and GitHub folder listings for the data step, each recorded
    (failures too) so a run replays without network, like `Searcher` and `EuropePMC`.

    With `http=None` it answers only from `replay`: no request and no DNS lookup (the URLs
    were checked when recorded); a download is read from the cache by its checksum.
    """

    def __init__(
        self,
        http: httpx.Client | None,
        replay: list[FetchCall] | None = None,
        dryad: DryadAuth | None = None,
    ) -> None:
        self.http = http
        self.dryad = dryad  # Dryad keys; without them Dryad records are rejected
        self._downloaded: dict[str, Download] = {}  # URL → this run's download of it
        self.replay = {(c.kind, c.url): c for c in replay or []}
        self.calls: list[FetchCall] = []

    @classmethod
    def from_replay(cls, calls: list[FetchCall]) -> "Fetcher":
        """A fetcher that answers only from a saved run's fetches; no network."""
        return cls(http=None, replay=calls)

    def download(
        self,
        url: str,
        cache_dir: Path,
        doi: str | None,
        license: str | None,
        suffix: str | None = None,
    ) -> Download:
        """`download()` live; in replay, the recorded file from `cache_dir`, checksum checked.
        A URL already downloaded in this run (to look inside a workbook while screening) is
        not fetched again."""
        if url in self._downloaded:
            return self._downloaded[url]
        if self.http is None:
            d = self._replayed("download", url).download
            assert d is not None
            path = cache_dir / Path(d.path).name
            if not path.exists():
                raise FileNotFoundError(f"{path} for {url} is not in the cache; re-record the run")
            if hashlib.sha256(path.read_bytes()).hexdigest() != d.sha256:
                raise ValueError(f"{path} does not match the recorded checksum {d.sha256}")
            return d.model_copy(update={"path": str(path)})
        http = self.http
        d = self._live(
            "download",
            url,
            lambda: download(url, http, cache_dir, doi, license, suffix=suffix, dryad=self.dryad),
        )
        self.calls[-1].download = d
        self._downloaded[url] = d
        return d

    def peek(self, url: str, suffix: str | None = None) -> list[str]:
        """`peek_columns()` live, or the recorded header."""
        if self.http is None:
            return self._replayed("peek", url).columns or []
        http = self.http
        columns = self._live(
            "peek", url, lambda: peek_columns(url, http, suffix=suffix, dryad=self.dryad)
        )
        self.calls[-1].columns = columns
        return columns

    def listing(self, owner: str, repo: str, ref: str, path: str) -> list[ListedFile]:
        """A named GitHub repo folder's entries, from the contents API or the recording."""
        url = f"{GITHUB_API}/{owner}/{repo}/contents/{path}?ref={ref}"
        if self.http is None:
            return self._replayed("listing", url).files or []
        http = self.http

        def get() -> list[ListedFile]:
            response = http.get(url)
            response.raise_for_status()
            return [ListedFile(**pick_listed(f)) for f in response.json()]

        files = self._live("listing", url, get)
        self.calls[-1].files = files
        return files

    def record(self, url: str) -> list[RepositoryFile] | None:
        """The files of the Zenodo, figshare, Dataverse or Dryad record `url` points to, from
        the site's metadata API or the recording; None (and no request) for any other link.
        A Dryad record without Dryad keys is rejected (and recorded) before any request."""
        api = record_api(url)
        if api is None:
            return None
        if self.http is None:
            return self._replayed("record", api).record or []
        http = self.http

        def get() -> list[RepositoryFile]:
            if is_dryad(api) and self.dryad is None:
                raise DownloadRejected(
                    "Dryad needs API keys: set DRYAD_CLIENT_ID and DRYAD_CLIENT_SECRET in .env"
                )
            return fetch_record(api, http, self.dryad)

        files = self._live("record", api, get)
        self.calls[-1].record = files
        return files

    def _live(self, kind: FetchKind, url: str, fetch: Callable[[], T]) -> T:
        """Run `fetch`, recording the call; a failure is recorded, then raised."""
        start = time.perf_counter()
        call = FetchCall(kind=kind, url=url, latency_s=0.0)
        self.calls.append(call)
        try:
            return fetch()
        except DownloadRejected as e:
            call.error, call.error_type = str(e), "rejected"
            raise
        except httpx.HTTPError as e:
            call.error, call.error_type = str(e), "http"
            raise
        finally:
            call.latency_s = time.perf_counter() - start

    def _replayed(self, kind: FetchKind, url: str) -> FetchCall:
        """The recorded call, re-raising its recorded failure; a miss raises `KeyError`."""
        if (kind, url) not in self.replay:
            raise KeyError(f"no recorded {kind} of {url!r}")
        call = self.replay[(kind, url)]
        self.calls.append(call)
        if call.error_type == "rejected":
            raise DownloadRejected(call.error)
        if call.error_type == "http":
            raise httpx.HTTPError(call.error or "")
        return call


def pick_listed(entry: dict) -> dict:
    """The fields of a GitHub contents API entry that `ListedFile` keeps."""
    return {k: entry[k] for k in ("name", "type", "size", "download_url")}
