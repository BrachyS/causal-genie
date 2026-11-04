"""Fetch a dataset file safely: allowlisted hosts, data formats only, size-capped, public IPs only.

Downloads are data: nothing in them is ever executed, and they live under cache/ (never
committed). Each file is recorded with its checksum so estimates trace to exact bytes.
"""

import hashlib
import ipaddress
import socket
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit

import httpx

from causal_genie.agent.state import Download
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


class DownloadRejected(ValueError):
    """The URL or response breaks a download rule; the caller may try the next candidate."""


def resolve_ips(host: str) -> list[str]:
    """Every IP `host` resolves to, so a public name pointing at a private network is caught."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, 443)]


def check_url(url: str) -> None:
    """Raise `DownloadRejected` unless `url` is https, allowlisted and resolves to public IPs."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise DownloadRejected(f"https only: {url}")
    host = parts.hostname or ""
    if not any(parts.path.startswith(p) for p in ALLOWLIST.get(host, ())):
        raise DownloadRejected(f"not on the allowlist: {url}")
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
