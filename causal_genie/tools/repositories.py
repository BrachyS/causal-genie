"""Repository record pages → their files, via each site's metadata API (T14.1).

A search hit or a paper's data link often points at a dataset's landing page (or its DOI),
not at a file. For Zenodo, figshare, Harvard Dataverse and Dryad this maps that link to the
site's read-only metadata API and lists the record's files with their license and DOI. Only
the API URL is built here, from an ID matched in the link; files still download through
`download()`'s allowlist.

Dryad's API and file downloads need a bearer token from the account's client ID and secret
(`DRYAD_CLIENT_ID`, `DRYAD_CLIENT_SECRET` in `.env`); `DryadAuth` gets and renews it. The
token is sent only to datadryad.org and is never logged or recorded on a run.

Not covered: OSF (files sit in nested folders and download from regional storage hosts).
"""

import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import PurePosixPath
from urllib.parse import quote, unquote

import httpx

from causal_genie.agent.state import RepositoryFile

log = logging.getLogger(__name__)

ZENODO = re.compile(
    r"https://(?:zenodo\.org/records?/|(?:dx\.)?doi\.org/10\.5281/zenodo\.)(\d+)/?(?:[?#].*)?",
    re.IGNORECASE,
)
FIGSHARE = re.compile(
    r"https://(?:(?:[\w-]+\.)?figshare\.com/articles/(?:[^/?#]+/)+?(\d{5,})(?:/\d+)?"
    r"|(?:dx\.)?doi\.org/10\.6084/m9\.figshare\.(\d+)(?:\.v\d+)?)/?(?:[?#].*)?",
    re.IGNORECASE,
)
DATAVERSE = re.compile(
    r"https://(?:dataverse\.harvard\.edu/dataset\.xhtml\?persistentId=doi:10\.7910/"
    r"|(?:dx\.)?doi\.org/10\.7910/)(DVN/[A-Z0-9]+)/?(?:[&#].*)?",
    re.IGNORECASE,
)
DRYAD = re.compile(
    r"https://(?:datadryad\.org/(?:stash/)?dataset/doi:|(?:dx\.)?doi\.org/)10\.5061/"
    r"(dryad\.[a-z0-9]+)/?(?:[?#].*)?",
    re.IGNORECASE,
)
DATAVERSE_API = "https://dataverse.harvard.edu/api"
DRYAD_HOST = "datadryad.org"
DRYAD_API = f"https://{DRYAD_HOST}/api/v2"
DRYAD_TOKEN_URL = f"https://{DRYAD_HOST}/oauth/token"
DRYAD_RENEW_MARGIN_S = 300  # get a new token this long before the old one expires
DRYAD_MAX_PAGES = 20  # a file list longer than this many pages is cut (and logged)
# Dataverse serves ingested tables as tab-separated `.tab` files.
FORMAT_ALIASES = {".tab": ".tsv"}


def record_api(url: str) -> str | None:
    """The metadata API URL for a Zenodo, figshare or Dataverse record page or DOI link; None
    for any other link."""
    if m := ZENODO.fullmatch(url):
        return f"https://zenodo.org/api/records/{m[1]}"
    if m := FIGSHARE.fullmatch(url):
        return f"https://api.figshare.com/v2/articles/{m[1] or m[2]}"
    if m := DATAVERSE.fullmatch(url):
        return f"{DATAVERSE_API}/datasets/:persistentId/?persistentId=doi:10.7910/{m[1].upper()}"
    # Search results can give Dryad links percent-encoded twice (`doi%253A10.5061%252F…`).
    if m := DRYAD.fullmatch(unquote(unquote(url))):
        return f"{DRYAD_API}/datasets/doi%3A10.5061%2F{m[1].lower()}"
    return None


def is_dryad(api_url: str) -> bool:
    """Whether a metadata API URL from `record_api` is Dryad's (which needs a token)."""
    return api_url.startswith(DRYAD_API)


class DryadAuth:
    """Bearer tokens for Dryad, from the account's client ID and secret. A token lasts about
    10 hours; it is reused until `DRYAD_RENEW_MARGIN_S` before it expires, then renewed."""

    def __init__(
        self, client_id: str, client_secret: str, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._clock = clock
        self._token: str | None = None
        self._expires_at = 0.0

    @classmethod
    def from_env(cls) -> "DryadAuth | None":
        """Keys from `DRYAD_CLIENT_ID` and `DRYAD_CLIENT_SECRET`; None if either is unset."""
        client_id, secret = os.getenv("DRYAD_CLIENT_ID"), os.getenv("DRYAD_CLIENT_SECRET")
        return cls(client_id, secret) if client_id and secret else None

    def headers(self, http: httpx.Client) -> dict[str, str]:
        """The Authorization header for a request to datadryad.org, renewing the token first
        if it is missing or about to expire."""
        if self._token is None or self._clock() >= self._expires_at - DRYAD_RENEW_MARGIN_S:
            response = http.post(
                DRYAD_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
            )
            response.raise_for_status()
            body = response.json()
            self._token = body["access_token"]
            self._expires_at = self._clock() + body["expires_in"]
            log.info("dryad: new token, expires in %s s", body["expires_in"])
        return {"Authorization": f"Bearer {self._token}"}


def fetch_record(api_url: str, http: httpx.Client, dryad: DryadAuth | None) -> list[RepositoryFile]:
    """A record's files from its metadata API: one request, or for Dryad the dataset then its
    latest version's file list, page by page, each with the token (`dryad` must be given)."""
    if not is_dryad(api_url):
        response = http.get(api_url)
        response.raise_for_status()
        return parse_record(api_url, response.json())
    assert dryad is not None, "a Dryad record needs DryadAuth"

    def get(url: str) -> dict:
        response = http.get(url, headers=dryad.headers(http))
        response.raise_for_status()
        return response.json()

    dataset = get(api_url)
    entries: list[dict] = []
    page: str | None = f"{dataset['_links']['stash:version']['href']}/files"
    for _ in range(DRYAD_MAX_PAGES):
        if page is None:
            break
        body = get(f"https://{DRYAD_HOST}{page}")
        entries += body.get("_embedded", {}).get("stash:files", [])
        page = body.get("_links", {}).get("next", {}).get("href")
    else:
        if page is not None:
            log.warning(
                "dryad: %s has over %d pages of files; kept the first", api_url, DRYAD_MAX_PAGES
            )
    return dryad_files(dataset, entries)


def dryad_files(dataset: dict, entries: list[dict]) -> list[RepositoryFile]:
    """Files of a Dryad dataset's latest version; each downloads from the files API."""
    return [
        RepositoryFile(
            name=f["path"],
            url=f"https://{DRYAD_HOST}{f['_links']['stash:download']['href']}",
            format=file_format(f["path"]),
            bytes=f.get("size"),
            license=dataset.get("license"),
            doi=dataset["identifier"].removeprefix("doi:"),
            record_title=dataset["title"],
        )
        for f in entries
    ]


def file_format(name: str) -> str:
    """The table format a file name says, lowercased (`x.CSV` → `.csv`, `x.tab` → `.tsv`)."""
    suffix = PurePosixPath(name).suffix.lower()
    return FORMAT_ALIASES.get(suffix, suffix)


def parse_record(api_url: str, body: dict) -> list[RepositoryFile]:
    """The files a record's metadata API response lists, each with where it downloads from."""
    if api_url.startswith("https://zenodo.org/"):
        return zenodo_files(body)
    if api_url.startswith("https://api.figshare.com/"):
        return figshare_files(body)
    return dataverse_files(body["data"])


def zenodo_files(body: dict) -> list[RepositoryFile]:
    """Files of a Zenodo record; each downloads from the record page's `files/<name>` link,
    which keeps the file name (and so its format) in the URL."""
    meta = body["metadata"]
    license = (meta.get("license") or {}).get("id")
    return [
        RepositoryFile(
            name=f["key"],
            url=f"https://zenodo.org/records/{body['id']}/files/{quote(f['key'])}?download=1",
            format=file_format(f["key"]),
            bytes=f.get("size"),
            license=license,
            doi=body.get("doi"),
            record_title=meta["title"],
        )
        for f in body.get("files", [])
    ]


def figshare_files(body: dict) -> list[RepositoryFile]:
    """Files of a figshare article. A link-only file points to another site, so it is left out."""
    license = (body.get("license") or {}).get("name")
    return [
        RepositoryFile(
            name=f["name"],
            url=f["download_url"],
            format=file_format(f["name"]),
            bytes=f.get("size"),
            license=license,
            doi=body.get("doi"),
            record_title=body["title"],
        )
        for f in body.get("files", [])
        if not f.get("is_link_only")
    ]


def dataverse_files(data: dict) -> list[RepositoryFile]:
    """Files of a Dataverse dataset's latest version, from the access API. Restricted files
    need the data holder's permission, so they are left out. Older datasets give the license
    as a plain string."""
    version = data["latestVersion"]
    license = version.get("license")
    if isinstance(license, dict):
        license = license.get("name")
    doi = data.get("persistentUrl", "").removeprefix("https://doi.org/") or None
    title = next(
        (
            f["value"]
            for f in version.get("metadataBlocks", {}).get("citation", {}).get("fields", [])
            if f.get("typeName") == "title"
        ),
        doi or "",
    )
    return [
        RepositoryFile(
            name=f["dataFile"]["filename"],
            url=f"{DATAVERSE_API}/access/datafile/{f['dataFile']['id']}",
            format=file_format(f["dataFile"]["filename"]),
            bytes=f["dataFile"].get("filesize"),
            license=license,
            doi=doi,
            record_title=title,
        )
        for f in version.get("files", [])
        if not f.get("restricted")
    ]
