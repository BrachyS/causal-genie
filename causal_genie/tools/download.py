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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, TypeVar
from urllib.parse import urljoin, urlsplit

import httpx

from causal_genie.agent.state import Download, FetchCall, ListedFile
from causal_genie.log import log_step

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
}
FORMATS = {".csv", ".tsv", ".xlsx", ".parquet", ".zip"}
MAX_BYTES = 50 * 1024 * 1024
MAX_REDIRECTS = 5
PEEK_BYTES = 64 * 1024  # a header line longer than this is not a table we can use
GITHUB_API = "https://api.github.com/repos"
T = TypeVar("T")


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


def download(
    url: str,
    http: httpx.Client,
    cache_dir: Path,
    doi: str | None,
    license: str | None,
    max_bytes: int = MAX_BYTES,
) -> Download:
    """Fetch `url` into `cache_dir` and return its provenance.

    Every redirect hop is re-checked against the allowlist and private-network rule.
    """
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    with log_step("download", url=url) as rec:
        if suffix not in FORMATS:
            raise DownloadRejected(f"format {suffix or '(none)'} not in {sorted(FORMATS)}: {url}")
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            check_url(current)
            with http.stream("GET", current, follow_redirects=False) as response:
                if response.is_redirect:
                    current = urljoin(current, response.headers["location"])
                    continue
                response.raise_for_status()
                body = read_capped(response, max_bytes)
                break
        else:
            raise DownloadRejected(f"more than {MAX_REDIRECTS} redirects: {url}")

        sha256 = hashlib.sha256(body).hexdigest()
        path = cache_dir / f"{sha256}{suffix}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        rec.update(sha256=sha256, bytes=len(body), final_url=current)
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


def peek_columns(url: str, http: httpx.Client, max_bytes: int = PEEK_BYTES) -> list[str]:
    """Column names from a CSV or TSV header line, so candidates can be scored before download.

    Reads at most `max_bytes` and keeps only the first line; rows are never kept. Other formats
    return []. Redirects are not followed (the full download re-checks them).
    """
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    if suffix not in {".csv", ".tsv"}:
        return []
    check_url(url)
    with log_step("peek", url=url) as rec, http.stream("GET", url) as response:
        response.raise_for_status()
        head = bytearray()
        for chunk in response.iter_bytes():
            head += chunk
            if b"\n" in head or len(head) > max_bytes:
                break
        if b"\n" not in head[:max_bytes]:
            raise DownloadRejected(f"no header line within {max_bytes} bytes: {url}")
        line = head[: head.index(b"\n")].decode("utf-8", errors="replace").rstrip("\r")
        columns = next(csv.reader([line], delimiter="\t" if suffix == ".tsv" else ","))
        rec["columns"] = columns
        return columns


class Fetcher:
    """Downloads, header reads and GitHub folder listings for the data step, each recorded
    (failures too) so a run replays without network, like `Searcher` and `EuropePMC`.

    With `http=None` it answers only from `replay`: no request and no DNS lookup (the URLs
    were checked when recorded); a download is read from the cache by its checksum.
    """

    def __init__(self, http: httpx.Client | None, replay: list[FetchCall] | None = None) -> None:
        self.http = http
        self.replay = {(c.kind, c.url): c for c in replay or []}
        self.calls: list[FetchCall] = []

    @classmethod
    def from_replay(cls, calls: list[FetchCall]) -> "Fetcher":
        """A fetcher that answers only from a saved run's fetches; no network."""
        return cls(http=None, replay=calls)

    def download(self, url: str, cache_dir: Path, doi: str | None, license: str | None) -> Download:
        """`download()` live; in replay, the recorded file from `cache_dir`, checksum checked."""
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
        d = self._live("download", url, lambda: download(url, http, cache_dir, doi, license))
        self.calls[-1].download = d
        return d

    def peek(self, url: str) -> list[str]:
        """`peek_columns()` live, or the recorded header."""
        if self.http is None:
            return self._replayed("peek", url).columns or []
        http = self.http
        columns = self._live("peek", url, lambda: peek_columns(url, http))
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

    def _live(
        self, kind: Literal["download", "peek", "listing"], url: str, fetch: Callable[[], T]
    ) -> T:
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

    def _replayed(self, kind: Literal["download", "peek", "listing"], url: str) -> FetchCall:
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
