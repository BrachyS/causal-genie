import json
from urllib.parse import parse_qsl

import httpx
import pytest
from conftest import FIXTURES

from causal_genie.tools.repositories import (
    DryadAuth,
    fetch_record,
    parse_record,
    record_api,
)

REPOS = FIXTURES / "repositories"


def fixture(name: str) -> dict:
    """A saved, trimmed response of a repository's public metadata API."""
    return json.loads((REPOS / name).read_text())


@pytest.mark.parametrize(
    ("page", "api"),
    [
        ("https://zenodo.org/records/7826073", "https://zenodo.org/api/records/7826073"),
        ("https://zenodo.org/record/7826073/", "https://zenodo.org/api/records/7826073"),
        ("https://doi.org/10.5281/zenodo.7826073", "https://zenodo.org/api/records/7826073"),
        (
            "https://figshare.com/articles/dataset/Data_from_Zomer_and_Ramsay/34341459",
            "https://api.figshare.com/v2/articles/34341459",
        ),
        (
            "https://monash.figshare.com/articles/dataset/plant_data/34341459/2",
            "https://api.figshare.com/v2/articles/34341459",
        ),
        (
            "https://doi.org/10.6084/m9.figshare.34341459.v1",
            "https://api.figshare.com/v2/articles/34341459",
        ),
        (
            "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SAWC6M",
            "https://dataverse.harvard.edu/api/datasets/:persistentId/"
            "?persistentId=doi:10.7910/DVN/SAWC6M",
        ),
        (
            "https://doi.org/10.7910/DVN/SAWC6M",
            "https://dataverse.harvard.edu/api/datasets/:persistentId/"
            "?persistentId=doi:10.7910/DVN/SAWC6M",
        ),
    ],
)
def test_record_pages_and_dois_map_to_the_metadata_api(page: str, api: str) -> None:
    assert record_api(page) == api


@pytest.mark.parametrize(
    "url",
    [
        "https://zenodo.org/communities/ecology",
        "https://zenodo.org/records/7826073/files/fig8_data.csv",  # a file, not the record
        "https://osf.io/xdpz5/",
        "https://doi.org/10.1038/s41586-020-2649-2",  # a paper
        "https://evil.example/records/7826073",
    ],
)
def test_other_links_are_not_records(url: str) -> None:
    assert record_api(url) is None


def test_zenodo_record_gives_each_file_with_license_and_doi() -> None:
    files = parse_record("https://zenodo.org/api/records/7826073", fixture("zenodo_record.json"))

    assert [(f.name, f.format, f.bytes) for f in files] == [
        ("fig5_data.nc", ".nc", 4584046),
        ("fig8_data.csv", ".csv", 626),
        ("fig1_simplified_data.csv", ".csv", 59887),
    ]
    assert files[1].url == "https://zenodo.org/records/7826073/files/fig8_data.csv?download=1"
    assert {(f.license, f.doi) for f in files} == {("cc-by-4.0", "10.5281/zenodo.7826073")}
    assert files[0].record_title == "Sensible heat fluxes control cloud trail strength"


def test_figshare_article_gives_download_links_and_formats_from_file_names() -> None:
    files = parse_record(
        "https://api.figshare.com/v2/articles/34341459", fixture("figshare_article.json")
    )

    csv = [f for f in files if f.format == ".csv"]
    assert [f.name for f in csv] == [
        "growth_form_raw.csv",
        "par_raw.csv",
        "site_information.csv",
        "soil_temperature_raw.csv",
        "vegetation_height_raw.csv",
    ]
    assert len(files) == 23  # README, a figure and photos too; the data step keeps tables only
    assert csv[0].url == "https://ndownloader.figshare.com/files/69927615"
    assert (csv[0].license, csv[0].doi) == ("CC BY 4.0", "10.6084/m9.figshare.34341459.v1")


def test_figshare_link_only_files_are_left_out() -> None:
    body = fixture("figshare_article.json")
    body["files"][2]["is_link_only"] = True

    names = [f.name for f in parse_record("https://api.figshare.com/v2/articles/1", body)]

    assert "growth_form_raw.csv" not in names


def test_dataverse_tab_file_is_read_as_tsv_from_the_access_api() -> None:
    api = (
        "https://dataverse.harvard.edu/api/datasets/:persistentId/"
        "?persistentId=doi:10.7910/DVN/SAWC6M"
    )
    (f,) = parse_record(api, fixture("dataverse_dataset.json"))

    assert (f.name, f.format, f.bytes) == ("experiment1_parasitoids.tab", ".tsv", 15418)
    assert f.url == "https://dataverse.harvard.edu/api/access/datafile/4461133"
    assert (f.license, f.doi) == ("CC0 1.0", "10.7910/DVN/SAWC6M")
    assert f.record_title == "static, low rate of parasitism"


def test_dataverse_restricted_files_are_left_out_and_old_string_licenses_read() -> None:
    body = fixture("dataverse_dataset.json")
    version = body["data"]["latestVersion"]
    version["license"] = "CC0"
    version["files"].append(
        {"restricted": True, "dataFile": {"id": 9, "filename": "secret.csv", "filesize": 1}}
    )
    api = "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/X"

    files = parse_record(api, body)

    assert [f.name for f in files] == ["experiment1_parasitoids.tab"]
    assert files[0].license == "CC0"


# --- Dryad: record lookups and downloads need a bearer token from the account's keys ---

DRYAD_API = "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.4b8gthtrz"
DRYAD_FILES = "https://datadryad.org/api/v2/versions/450332/files"
TOKEN = "tok-secret-1"


class Clock:
    """A settable clock for token expiry."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def dryad_web(pages: dict[str, dict] | None = None):
    """A fake Dryad: the token endpoint plus the saved dataset and files responses. Returns
    the client and the requests it saw."""
    served = {DRYAD_API: fixture("dryad_dataset.json"), DRYAD_FILES: fixture("dryad_files.json")}
    served |= pages or {}
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/oauth/token":
            form = dict(parse_qsl(request.content.decode()))
            assert form == {
                "grant_type": "client_credentials",
                "client_id": "id",
                "client_secret": "s",
            }
            return httpx.Response(200, json={"access_token": TOKEN, "expires_in": 35999})
        body = served.get(str(request.url))
        return httpx.Response(200, json=body) if body is not None else httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handle)), seen


@pytest.mark.parametrize(
    "page",
    [
        "https://datadryad.org/dataset/doi:10.5061/dryad.4b8gthtrz",
        "https://datadryad.org/stash/dataset/doi:10.5061/dryad.4b8gthtrz",
        "https://datadryad.org/dataset/doi%253A10.5061%252Fdryad.4b8gthtrz",  # as search gives it
        "https://doi.org/10.5061/dryad.4b8gthtrz",
    ],
)
def test_dryad_pages_and_dois_map_to_the_dataset_api(page: str) -> None:
    assert record_api(page) == DRYAD_API


def test_dryad_record_lists_files_with_token_license_and_doi() -> None:
    http, seen = dryad_web()
    auth = DryadAuth("id", "s", clock=Clock())

    files = fetch_record(DRYAD_API, http, auth)

    assert len(files) == 7
    f = next(f for f in files if f.name == "raw_location_data.csv")
    assert (f.url, f.format, f.bytes) == (
        "https://datadryad.org/api/v2/files/4839169/download",
        ".csv",
        1916,
    )
    assert (f.license, f.doi) == (
        "https://spdx.org/licenses/CC0-1.0.html",
        "10.5061/dryad.4b8gthtrz",
    )
    assert f.record_title == "Moʻorea plant herbivory damage type data"
    api_calls = [r for r in seen if r.url.path != "/oauth/token"]
    assert [str(r.url) for r in api_calls] == [DRYAD_API, DRYAD_FILES]
    assert {r.headers["authorization"] for r in api_calls} == {f"Bearer {TOKEN}"}
    assert sum(r.url.path == "/oauth/token" for r in seen) == 1


def test_dryad_file_list_follows_next_pages() -> None:
    first = fixture("dryad_files.json")
    second = {"_embedded": {"stash:files": [first["_embedded"]["stash:files"].pop()]}, "_links": {}}
    first["_links"]["next"] = {"href": "/api/v2/versions/450332/files?page=2"}
    http, _ = dryad_web({DRYAD_FILES: first, f"{DRYAD_FILES}?page=2": second})

    files = fetch_record(DRYAD_API, http, DryadAuth("id", "s", clock=Clock()))

    assert len(files) == 7


def test_dryad_token_is_reused_then_renewed_before_it_expires() -> None:
    http, seen = dryad_web()
    clock = Clock()
    auth = DryadAuth("id", "s", clock=clock)

    auth.headers(http)
    clock.now += 35999 - 600  # still more than the renewal margin left
    auth.headers(http)
    assert sum(r.url.path == "/oauth/token" for r in seen) == 1

    clock.now += 400  # within the margin: renew
    auth.headers(http)
    assert sum(r.url.path == "/oauth/token" for r in seen) == 2


def test_dryad_token_request_failure_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)))

    with pytest.raises(httpx.HTTPStatusError):
        DryadAuth("id", "bad", clock=Clock()).headers(http)


def test_dryad_keys_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRYAD_CLIENT_ID", raising=False)
    monkeypatch.delenv("DRYAD_CLIENT_SECRET", raising=False)
    assert DryadAuth.from_env() is None

    monkeypatch.setenv("DRYAD_CLIENT_ID", "id")
    monkeypatch.setenv("DRYAD_CLIENT_SECRET", "s")
    assert isinstance(DryadAuth.from_env(), DryadAuth)
