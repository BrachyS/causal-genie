"""EPA AirData pre-generated files as dataset candidates (T23.1).

EPA publishes US air-quality data as about 2,000 ready-made ZIP files at
aqs.epa.gov/aqsweb/airdata/, each holding one CSV (one row per monitor and day, or per county
and day for the AQI files). Search rarely finds a file among them, so the harness builds the
candidates itself from this short catalog, for the years the question names; the data step
keeps only those whose title shares a keyword with the question. US government work, so
public domain.

One year per file: a question comparing years gets one candidate per year, but the MVP
analyses one table, so stacking years waits for the multi-table step (T23.5).
"""

import re
from dataclasses import dataclass
from datetime import datetime

AIRDATA = "https://aqs.epa.gov/aqsweb/airdata"
FIRST_YEAR = 1980  # the first year AirData has files for
MAX_YEARS = 6  # a longer span keeps its latest years, so candidates stay few
SOURCE = "EPA AirData (US air quality, air pollution)"
# File stem → (what it measures, the unit and period of a row). A question is matched on the
# first part only, so `count` does not match "county" or `day` "daily".
CATALOG = {
    "daily_44201": ("ozone (O3)", "daily, by monitor"),
    "daily_88101": ("fine particulate matter (PM2.5)", "daily, by monitor"),
    "daily_81102": ("particulate matter (PM10)", "daily, by monitor"),
    "daily_42602": ("nitrogen dioxide (NO2)", "daily, by monitor"),
    "daily_42401": ("sulfur dioxide (SO2)", "daily, by monitor"),
    "daily_42101": ("carbon monoxide (CO)", "daily, by monitor"),
    "annual_conc_by_monitor": ("concentration of every pollutant", "annual, by monitor"),
    "daily_aqi_by_county": ("air quality index (AQI)", "daily, by county"),
    "annual_aqi_by_county": ("air quality index (AQI)", "annual, by county"),
}
YEAR = re.compile(r"\b(19[89]\d|20\d\d)\b")
SPAN = re.compile(r"\b(19[89]\d|20\d\d)\s*[-–—]\s*(19[89]\d|20\d\d)\b")


@dataclass(frozen=True)
class AirDataFile:
    url: str
    title: str
    description: str
    topic: str  # the words a question's keywords are matched against


def question_years(time_window: str | None, searched: datetime) -> list[int]:
    """The years to offer files for: those the question's time window names (a span like
    `2015–2020` counts every year in it), within AirData's range and not after `searched`
    (when the data search ran, as recorded, so replays agree); else the last full year
    before it."""
    window = time_window or ""
    years = {int(y) for y in YEAR.findall(window)}
    for start, end in SPAN.findall(window):
        years |= set(range(int(start), int(end) + 1))
    years = sorted(y for y in years if FIRST_YEAR <= y <= searched.year)
    return years[-MAX_YEARS:] or [searched.year - 1]


def airdata_files(years: list[int]) -> list[AirDataFile]:
    """Every catalog file for each year, with a title and description for the model's
    coverage scoring and a topic for keyword matching."""
    return [
        AirDataFile(
            url=f"{AIRDATA}/{stem}_{year}.zip",
            title=f"{SOURCE}: {what}, {unit}, {year}",
            description=(
                "US EPA, public domain. One CSV; columns include state, county and site "
                "codes, latitude, longitude, date and the measured value or AQI."
            ),
            topic=f"{SOURCE} {what}",
        )
        for year in years
        for stem, (what, unit) in CATALOG.items()
    ]
