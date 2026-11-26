from datetime import UTC, datetime

import pytest

from causal_genie.tools.airdata import MAX_YEARS, airdata_files, question_years

STARTED = datetime(2026, 11, 26, tzinfo=UTC)


@pytest.mark.parametrize(
    ("window", "years"),
    [
        ("2020", [2020]),
        ("March to May 2020", [2020]),
        ("2019 vs 2020", [2019, 2020]),
        ("2015–2020", [2015, 2016, 2017, 2018, 2019, 2020]),
        ("2015-2020", [2015, 2016, 2017, 2018, 2019, 2020]),
        (None, [2025]),  # no years given: the last full year before the run
        ("1970s", [2025]),  # before AirData starts (1980)
        ("2030", [2025]),  # in the future
    ],
)
def test_years_come_from_the_question_time_window(window: str | None, years: list[int]) -> None:
    assert question_years(window, STARTED) == years


def test_a_long_span_keeps_the_latest_years() -> None:
    years = question_years("1990-2020", STARTED)

    assert len(years) == MAX_YEARS and years[-1] == 2020


def test_each_year_lists_the_daily_pollutant_and_aqi_files() -> None:
    files = airdata_files([2020])

    urls = [f.url for f in files]
    assert "https://aqs.epa.gov/aqsweb/airdata/daily_42602_2020.zip" in urls
    assert "https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_2020.zip" in urls
    no2 = next(f for f in files if "42602" in f.url)
    assert "nitrogen dioxide (NO2)" in no2.title and "2020" in no2.title
    assert "EPA AirData" in no2.title and "air pollution" in no2.title
