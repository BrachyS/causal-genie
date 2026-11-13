import pandas as pd
import pytest

from causal_genie.agent.state import Mapping, NodeMapping
from causal_genie.tools.derive import (
    day_of_week,
    full_moon,
    node_table,
    season,
    to_dates,
    us_holiday,
    weekend,
)

# Total lunar eclipses happen at full moon; dates (UTC) from NASA's eclipse catalogue.
ECLIPSE_FULL_MOONS = [
    "2000-01-21", "2000-07-16", "2001-01-09", "2003-05-16", "2003-11-09", "2004-10-28",
    "2007-03-03", "2007-08-28", "2008-02-21", "2010-12-21", "2011-06-15", "2014-04-15",
    "2014-10-08",
]  # fmt: skip


def dates(*days: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(days)))


def test_full_moon_matches_published_eclipse_dates() -> None:
    flags = full_moon(dates(*ECLIPSE_FULL_MOONS))

    assert flags.tolist() == [1] * len(ECLIPSE_FULL_MOONS)


def test_full_moon_is_one_day_per_lunar_month() -> None:
    year = pd.Series(pd.date_range("2012-01-01", "2012-12-31"))

    flags = full_moon(year)

    assert flags.sum() == 13  # 2012 had a blue moon (Aug 2 and Aug 31)
    assert flags[year == pd.Timestamp("2012-08-31")].item() == 1
    assert flags[year == pd.Timestamp("2012-08-30")].item() == 0


def test_dates_from_year_month_day_columns_or_one_date_column() -> None:
    df = pd.DataFrame({"year": [2000], "month": [1], "date_of_month": [21], "d": ["2000-01-21"]})

    assert to_dates(df, ["year", "month", "date_of_month"]).tolist() == [pd.Timestamp("2000-01-21")]
    assert to_dates(df, ["d"]).tolist() == [pd.Timestamp("2000-01-21")]
    with pytest.raises(ValueError, match="date"):
        to_dates(df, ["year", "month"])


def test_calendar_rules() -> None:
    d = dates("2000-07-04", "2000-07-05", "2000-11-23", "2000-12-25", "2000-07-08")

    assert day_of_week(d).tolist() == ["Tue", "Wed", "Thu", "Mon", "Sat"]
    assert weekend(d).tolist() == [0, 0, 0, 0, 1]
    assert us_holiday(d).tolist() == [1, 0, 1, 1, 0]  # July 4, Thanksgiving, Christmas
    assert season(dates("2000-01-15", "2000-04-15", "2000-07-15", "2000-10-15")).tolist() == [
        "winter", "spring", "summer", "autumn",
    ]  # fmt: skip


def node(name: str, columns: list[str], derive=None, cutoff=None, categorical=False):
    return NodeMapping(
        node=name, columns=columns, derive=derive, cutoff=cutoff, categorical=categorical
    )


def test_node_table_builds_one_column_per_mapped_node() -> None:
    raw = pd.DataFrame(
        {
            "year": [2000, 2000],
            "month": [1, 1],
            "date_of_month": [21, 22],
            "day_of_week": [5, 6],
            "births": [11000, 9000],
            "pain": [0, 3],
        }
    )
    mapping = Mapping(
        nodes=[
            node("full_moon", ["year", "month", "date_of_month"], derive="full_moon"),
            node("birth", ["births"]),
            node("weekday", ["day_of_week"], categorical=True),
            node("sore_throat", ["pain"], derive="cutoff", cutoff=0),
            node("parity", []),  # not in the table
        ],
        unit=None,
        time=None,
    )

    table = node_table(raw, mapping)

    assert list(table.columns) == ["full_moon", "birth", "weekday", "sore_throat"]
    assert table["full_moon"].tolist() == [1, 0]
    assert table["weekday"].tolist() == ["5", "6"]
    assert table["sore_throat"].tolist() == [0, 1]


def test_node_table_keeps_panel_unit_and_time_columns() -> None:
    raw = pd.DataFrame({"city": ["a", "a"], "month": [1, 2], "launched": [0, 1]})
    mapping = Mapping(nodes=[node("launch", ["launched"])], unit="city", time="month")

    assert list(node_table(raw, mapping).columns) == ["launch", "city", "month"]


def test_cutoff_keeps_missing_values_missing() -> None:
    raw = pd.DataFrame({"pain": [0, 3, None]})
    mapping = Mapping(nodes=[node("sore", ["pain"], "cutoff", 0)], unit=None, time=None)

    assert node_table(raw, mapping)["sore"].tolist()[:2] == [0, 1]
    assert pd.isna(node_table(raw, mapping)["sore"].iloc[2])


def test_cutoff_over_several_columns_is_yes_if_any_is_above() -> None:
    """E.g. any sore throat in 24 h from pain scores at 4 time points."""
    raw = pd.DataFrame(
        {
            "pain_30m": [0, 0, 2, None, 0, None],
            "pain_4h": [0, 1, None, 0, None, None],
        }
    )
    mapping = Mapping(
        nodes=[node("sore", ["pain_30m", "pain_4h"], "cutoff", 0)], unit=None, time=None
    )

    sore = node_table(raw, mapping)["sore"]

    # yes if any score is above; no only if every score is there and none is above
    assert sore.iloc[:3].tolist() == [0, 1, 1]
    assert sore.iloc[3:].isna().all()
