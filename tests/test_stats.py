import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES
from scipy import stats as sps
from statsmodels.stats.proportion import confint_proportions_2indep

from causal_genie.tools.stats import backdoor_effect, mean_difference, rct_effect, risk_difference

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


# --- backdoor_effect (T9.5) -------------------------------------------------------------------


def confounded(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Age raises both the chance of exposure and the outcome; the true effect of x is 2."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    region = rng.choice(["north", "south", "west"], n)
    x = (rng.random(n) < 1 / (1 + np.exp(-(age - 50) / 5))).astype(int)
    y = 2 * x + 0.3 * age + np.where(region == "south", 4, 0) + rng.normal(0, 3, n)
    return pd.DataFrame({"x": x, "y": y, "age": age, "region": region})


def hc1_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and HC1 robust standard errors, computed by hand."""
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    resid = y - X @ beta
    meat = X.T @ (X * resid[:, None] ** 2)
    cov = bread @ meat @ bread * n / (n - k)
    return beta, np.sqrt(np.diag(cov))


def test_backdoor_effect_matches_hand_ols_with_hc1_errors() -> None:
    df = confounded()
    X = np.column_stack([np.ones(len(df)), df["x"], df["age"]])
    beta, se = hc1_ols(X, df["y"].to_numpy())
    z = sps.norm.ppf(0.975)

    r = backdoor_effect(df, "x", "y", adjust_for=["age"], scale="mean_difference")

    assert r["estimate"] == pytest.approx(beta[1])
    assert (r["ci_low"], r["ci_high"]) == pytest.approx((beta[1] - z * se[1], beta[1] + z * se[1]))
    assert r["n"] == len(df)
    assert r["adjusted_for"] == ["age"]


def test_backdoor_effect_recovers_the_true_effect_the_naive_one_misses() -> None:
    r = backdoor_effect(
        confounded(), "x", "y", adjust_for=["age", "region"], scale="mean_difference"
    )

    assert r["ci_low"] < 2 < r["ci_high"]
    assert r["naive_ci_low"] > 2  # older people are exposed more and score higher
    assert r["naive_estimate"] == pytest.approx(
        confounded().query("x == 1")["y"].mean() - confounded().query("x == 0")["y"].mean()
    )


def test_backdoor_effect_without_covariates_is_the_difference_in_proportions() -> None:
    d3 = pd.read_csv(FIXTURES / "d3_onboarding_ab.csv")
    rct = rct_effect(d3, "new_flow", "activated_7d", "risk_difference")

    r = backdoor_effect(d3, "new_flow", "activated_7d", adjust_for=[], scale="risk_difference")

    assert r["estimate"] == pytest.approx(rct["estimate"])
    assert r["ci_low"] < 0.05 < r["ci_high"]  # D3's true effect


def test_backdoor_odds_ratio_without_covariates_matches_the_two_by_two_table() -> None:
    a, b, c, d = 30, 70, 15, 85  # exposed: 30 events of 100; unexposed: 15 of 100
    df = pd.DataFrame({"x": [1] * 100 + [0] * 100, "y": [1] * a + [0] * b + [1] * c + [0] * d})
    log_or, se = np.log(a * d / (b * c)), np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)  # Woolf
    z = sps.norm.ppf(0.975)

    r = backdoor_effect(df, "x", "y", adjust_for=[], scale="odds_ratio")

    assert r["estimate"] == pytest.approx(np.exp(log_or))
    assert (r["ci_low"], r["ci_high"]) == pytest.approx(
        (np.exp(log_or - z * se), np.exp(log_or + z * se)), rel=1e-4
    )


def test_backdoor_effect_rejects_missing_values_and_non_binary_exposure() -> None:
    df = confounded().head(10)
    df.loc[0, "age"] = None
    with pytest.raises(ValueError, match="missing"):
        backdoor_effect(df, "x", "y", adjust_for=["age"], scale="mean_difference")

    df = confounded().head(10).assign(x=range(10))
    with pytest.raises(ValueError, match="0/1"):
        backdoor_effect(df, "x", "y", adjust_for=[], scale="mean_difference")
