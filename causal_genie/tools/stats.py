"""Effect estimators. Every number in a report comes from one of these tested functions.

Effects are always treated minus control, so a negative risk difference means the treatment
lowers the outcome's risk.
"""

from collections.abc import Sequence
from math import sqrt
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def wilson_interval(events: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for one proportion; stays inside [0, 1] even at 0 or n events."""
    p = events / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z / (1 + z**2 / n) * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return center - half, center + half


def risk_difference(
    events_treated: int, n_treated: int, events_control: int, n_control: int, level: float = 0.95
) -> dict:
    """Risk difference with Newcombe's hybrid score CI (method 10, Newcombe 1998).

    Chosen over the Wald interval because it behaves at small counts and zero-event arms.
    """
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    p1, p2 = events_treated / n_treated, events_control / n_control
    l1, u1 = wilson_interval(events_treated, n_treated, z)
    l2, u2 = wilson_interval(events_control, n_control, z)
    d = p1 - p2
    return {
        "scale": "risk_difference",
        "estimate": d,
        "ci_low": d - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
        "ci_high": d + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
        "level": level,
        "method": "Newcombe hybrid score",
        "events_treated": events_treated,
        "n_treated": n_treated,
        "risk_treated": p1,
        "events_control": events_control,
        "n_control": n_control,
        "risk_control": p2,
    }


def mean_difference(
    treated: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    level: float = 0.95,
) -> dict:
    """Difference in means with a Welch CI (no equal-variance assumption)."""
    t, c = np.asarray(treated, dtype=float), np.asarray(control, dtype=float)
    v_t, v_c = t.var(ddof=1) / len(t), c.var(ddof=1) / len(c)
    se = sqrt(v_t + v_c)
    df = (v_t + v_c) ** 2 / (v_t**2 / (len(t) - 1) + v_c**2 / (len(c) - 1))
    d = float(t.mean() - c.mean())
    half = float(stats.t.ppf(1 - (1 - level) / 2, df)) * se
    return {
        "scale": "mean_difference",
        "estimate": d,
        "ci_low": d - half,
        "ci_high": d + half,
        "level": level,
        "method": "Welch",
        "n_treated": len(t),
        "mean_treated": float(t.mean()),
        "n_control": len(c),
        "mean_control": float(c.mean()),
    }


def rct_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    scale: Literal["risk_difference", "mean_difference"],
    level: float = 0.95,
) -> dict:
    """Effect of randomized `treatment` (0/1 column) on `outcome`, analyzed as randomized.

    Missing values are an error, not silently dropped: the caller decides how to handle them
    and the report must say how many rows were excluded.
    """
    data = df[[treatment, outcome]]
    if data.isna().any().any():
        raise ValueError(f"missing values in {treatment!r} or {outcome!r}; drop them explicitly")
    if set(data[treatment].unique()) != {0, 1}:
        raise ValueError(
            f"treatment column {treatment!r} must be 0/1 with both exposed and unexposed rows; "
            f"got values {sorted(data[treatment].unique())}"
        )
    treated = data.loc[data[treatment] == 1, outcome]
    control = data.loc[data[treatment] == 0, outcome]
    if scale == "risk_difference":
        if not set(data[outcome].unique()) <= {0, 1}:
            raise ValueError(f"outcome column {outcome!r} must be 0/1 for a risk difference")
        return risk_difference(
            int(treated.sum()), len(treated), int(control.sum()), len(control), level
        )
    return mean_difference(treated.to_numpy(), control.to_numpy(), level)


def backdoor_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjust_for: list[str],
    scale: Literal["risk_difference", "mean_difference", "odds_ratio"],
    level: float = 0.95,
    cluster: str | None = None,
) -> dict:
    """Effect of a 0/1 `treatment` on `outcome` adjusting for the backdoor set `adjust_for`,
    with the unadjusted (naive) estimate alongside so the report can show what adjustment did.

    Risk and mean differences come from OLS (a linear probability model for a 0/1 outcome),
    odds ratios from logistic regression; both use HC1 robust standard errors, or
    cluster-robust ones grouped by the `cluster` column when rows are not independent (e.g.
    many states on one date share that date's exposure). Text columns become one dummy per
    category. Missing values are an error, as in `rct_effect`.
    """
    data = df[[treatment, outcome, *adjust_for]]
    groups = None if cluster is None else pd.factorize(df[cluster])[0]
    if data.isna().any().any():
        raise ValueError(f"missing values in {list(data.columns)}; drop them explicitly")
    if set(data[treatment].unique()) != {0, 1}:
        raise ValueError(
            f"treatment column {treatment!r} must be 0/1 with both exposed and unexposed rows; "
            f"got values {sorted(data[treatment].unique())}"
        )
    if scale != "mean_difference" and not set(data[outcome].unique()) <= {0, 1}:
        raise ValueError(f"outcome column {outcome!r} must be 0/1 for {scale}")
    covariates = (
        pd.get_dummies(data[adjust_for], drop_first=True, dtype=float)
        if adjust_for
        else pd.DataFrame(index=data.index)
    )
    z = float(stats.norm.ppf(1 - (1 - level) / 2))

    def fit(columns: pd.DataFrame) -> tuple[float, float, float]:
        """Treatment coefficient and its CI on the reported scale."""
        X = sm.add_constant(pd.concat([data[[treatment]].astype(float), columns], axis=1))
        y = data[outcome].astype(float)
        # cov_kwds only when clustering: Logit warns (and will raise) on cov_kwds=None
        cluster_kwds = {"groups": groups}
        if scale == "odds_ratio" and groups is None:
            model = sm.Logit(y, X).fit(disp=0, cov_type="HC1")
        elif scale == "odds_ratio":
            model = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds=cluster_kwds)
        elif groups is None:
            model = sm.OLS(y, X).fit(cov_type="HC1")
        else:
            model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds=cluster_kwds)
        b, se = float(model.params[treatment]), float(model.bse[treatment])
        low, high = b - z * se, b + z * se
        if scale == "odds_ratio":
            return float(np.exp(b)), float(np.exp(low)), float(np.exp(high))
        return b, low, high

    estimate, ci_low, ci_high = fit(covariates)
    naive, naive_low, naive_high = fit(covariates.iloc[:, :0])
    return {
        "scale": scale,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "level": level,
        "method": ("logistic" if scale == "odds_ratio" else "OLS")
        + (", HC1 robust SEs" if cluster is None else f", cluster-robust SEs by {cluster}"),
        "n": len(data),
        "n_treated": int(data[treatment].sum()),
        "adjusted_for": list(adjust_for),
        "naive_estimate": naive,
        "naive_ci_low": naive_low,
        "naive_ci_high": naive_high,
    }
