"""Compute DAG nodes the table doesn't hold directly (calendar, moon phase, a cutoff on a score).

The mapping names a rule from a fixed menu (`DeriveRule`) and the columns it reads; nothing a
model writes is ever executed. Dates are calendar days; the full moon is the UTC day that holds
the full-moon instant, as published moon calendars list it.
"""

import logging

import ephem
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from causal_genie.agent.state import Mapping, NodeMapping

log = logging.getLogger(__name__)

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASONS = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def to_dates(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Calendar dates from one date column, or from year, month and day columns (in that order).

    An impossible date (e.g. Feb 29 in a non-leap year, found in mosaicData::Birthdays) is
    missing, and logged with a count, so its row is excluded and counted like any missing value.
    """
    if len(columns) == 1:
        dates = pd.to_datetime(df[columns[0]], errors="coerce").dt.normalize()
    elif len(columns) == 3:
        year, month, day = columns
        parts = pd.DataFrame({"year": df[year], "month": df[month], "day": df[day]})
        dates = pd.to_datetime(parts, errors="coerce")
    else:
        raise ValueError(f"a date needs one date column or year, month, day; got {columns}")
    if dates.dt.tz is not None:  # "1969-01-01T00:00:00Z": the calendar day, without the zone
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    if invalid := int((dates.isna() & df[columns].notna().all(axis=1)).sum()):
        log.warning("%d rows have an invalid date in %s; treated as missing", invalid, columns)
    return dates


def day_of_week(dates: pd.Series) -> pd.Series:
    """Day names, so models adjust for each day as a category, not a number."""
    return dates.dt.day_of_week.map(dict(enumerate(DAYS)))


def weekend(dates: pd.Series) -> pd.Series:
    return (dates.dt.day_of_week >= 5).astype(int)


def us_holiday(dates: pd.Series) -> pd.Series:
    """US federal holidays, on the day they are observed."""
    holidays = USFederalHolidayCalendar().holidays(dates.min(), dates.max())
    return dates.isin(holidays).astype(int)


def season(dates: pd.Series) -> pd.Series:
    """Northern-hemisphere meteorological season."""
    return dates.dt.month.map(SEASONS)


def full_moon(dates: pd.Series) -> pd.Series:
    """1 on the UTC day that holds a full-moon instant (one day per lunar month), else 0."""
    moon = ephem.previous_full_moon(ephem.Date(dates.min().to_pydatetime()))
    days = set()
    while moon.datetime().date() <= dates.max().date():
        days.add(pd.Timestamp(moon.datetime().date()))
        moon = ephem.next_full_moon(moon + 1)
    return dates.isin(days).astype(int)


DATE_RULES = {
    "day_of_week": day_of_week,
    "weekend": weekend,
    "holiday": us_holiday,
    "season": season,
    "full_moon": full_moon,
}


def node_values(df: pd.DataFrame, m: NodeMapping) -> pd.Series:
    """One node's values: the column as is, a date rule, or a 0/1 cutoff on one or more
    scores (yes if any is above it, e.g. sore throat at any of several time points)."""
    if m.derive in DATE_RULES:
        dates = to_dates(df, m.columns)
        return DATE_RULES[m.derive](dates).where(dates.notna())
    if m.derive == "cutoff":
        assert m.cutoff is not None
        scores = df[m.columns]
        above = (scores > m.cutoff).any(axis=1)
        # yes if any score is above; no only if every score is there; else missing
        return above.astype(int).where(above | scores.notna().all(axis=1))
    column = df[m.columns[0]]
    return column.astype(str).where(column.notna()) if m.categorical else column


def node_table(df: pd.DataFrame, mapping: Mapping) -> pd.DataFrame:
    """The analysis table: one column per mapped node (named after it), plus a panel's unit
    and time columns. Unmapped nodes are left out; missing values stay missing."""
    table = pd.DataFrame({m.node: node_values(df, m) for m in mapping.nodes if m.columns})
    for column in (mapping.unit, mapping.time):
        if column is not None:
            table[column] = df[column]
    return table
