import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from statsmodels.stats.proportion import confint_proportions_2indep

from causal_genie.tools.stats import mean_difference, rct_effect, risk_difference

# Newcombe (1998) Stat Med 17:873, Table II, method 10 (hybrid score): published to 4 decimals.
NEWCOMBE_TABLE = [
    ((56, 70, 48, 80), (0.0524, 0.3339)),
    ((9, 10, 3, 10), (0.1705, 0.8090)),
    ((6, 7, 2, 7), (0.0582, 0.8062)),
    ((5, 56, 0, 29), (-0.0381, 0.1926)),  # zero-event control arm
    ((0, 10, 0, 20), (-0.1611, 0.2775)),  # zero events in both arms
    ((10, 10, 0, 20), (0.6791, 1.0000)),  # all events vs none
]


@pytest.mark.parametrize(("counts", "published"), NEWCOMBE_TABLE)
def test_risk_difference_matches_newcombe_paper(counts, published) -> None:
    r = risk_difference(*counts)

    assert r["estimate"] == pytest.approx(counts[0] / counts[1] - counts[2] / counts[3])
    assert (r["ci_low"], r["ci_high"]) == pytest.approx(published, abs=1e-4)


@pytest.mark.parametrize(("counts", "_"), NEWCOMBE_TABLE)
def test_risk_difference_matches_statsmodels(counts, _) -> None:
    low, high = confint_proportions_2indep(*counts, method="newcomb", compare="diff")

    r = risk_difference(*counts)

    assert (r["ci_low"], r["ci_high"]) == pytest.approx((low, high), abs=1e-4)


def test_mean_difference_matches_scipy_welch() -> None:
    rng = np.random.default_rng(0)
    treated, control = rng.normal(5, 2, 40), rng.normal(4, 3, 55)
    welch = sps.ttest_ind(treated, control, equal_var=False).confidence_interval(0.95)

    r = mean_difference(treated, control)

    assert r["estimate"] == pytest.approx(treated.mean() - control.mean())
    assert (r["ci_low"], r["ci_high"]) == pytest.approx((welch.low, welch.high), abs=1e-4)


def test_mean_difference_matches_hand_calculation() -> None:
    # treated mean 3, var 1; control mean 1, var 1; se = sqrt(1/3 + 1/3); Welch df = 4
    r = mean_difference([2, 3, 4], [0, 1, 2])

    half = sps.t.ppf(0.975, 4) * np.sqrt(2 / 3)
    assert (r["estimate"], r["ci_low"], r["ci_high"]) == pytest.approx((2, 2 - half, 2 + half))


def test_rct_effect_risk_difference_counts_each_arm() -> None:
    df = pd.DataFrame({"arm": [1] * 4 + [0] * 4, "y": [1, 1, 1, 0, 1, 0, 0, 0]})

    r = rct_effect(df, treatment="arm", outcome="y", scale="risk_difference")

    assert (r["events_treated"], r["n_treated"], r["events_control"], r["n_control"]) == (
        3,
        4,
        1,
        4,
    )
    assert r["estimate"] == pytest.approx(0.5)


def test_rct_effect_mean_difference() -> None:
    df = pd.DataFrame({"arm": [1, 1, 1, 0, 0, 0], "y": [2, 3, 4, 0, 1, 2]})

    r = rct_effect(df, treatment="arm", outcome="y", scale="mean_difference")

    assert (r["n_treated"], r["n_control"], r["estimate"]) == (3, 3, pytest.approx(2))


def test_rct_effect_rejects_missing_values() -> None:
    df = pd.DataFrame({"arm": [1, 1, 0, 0], "y": [1, None, 0, 1]})

    with pytest.raises(ValueError, match="missing"):
        rct_effect(df, treatment="arm", outcome="y", scale="risk_difference")


def test_rct_effect_rejects_non_binary_treatment() -> None:
    df = pd.DataFrame({"arm": [0, 1, 2], "y": [0, 1, 1]})

    with pytest.raises(ValueError, match="0/1"):
        rct_effect(df, treatment="arm", outcome="y", scale="risk_difference")
