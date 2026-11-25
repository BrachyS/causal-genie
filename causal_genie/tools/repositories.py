"""Repository record pages → their files, via each site's public metadata API (T14.1).

A search hit or a paper's data link often points at a dataset's landing page (or its DOI),
not at a file. For Zenodo, figshare and Harvard Dataverse this maps that link to the site's
read-only metadata API and lists the record's files with their license and DOI. Only the
API URL is built here, from an ID matched in the link; files still download through
`download()`'s allowlist.

Not covered: Dryad (file downloads need an API token) and OSF (files sit in nested folders
and download from regional storage hosts).
"""

import re
from pathlib import PurePosixPath
from urllib.parse import quote

from causal_genie.agent.state import RepositoryFile

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
DATAVERSE_API = "https://dataverse.harvard.edu/api"
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
    return None


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
