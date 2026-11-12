"""The synthetic fixtures (T9.6, T9.4.1) have the designs they claim, so design and estimate
tests built on them test the tools, not the data."""

import pandas as pd
from conftest import FIXTURES


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name)


def test_d3_is_the_full_randomized_roster() -> None:
    d3 = read("d3_onboarding_ab.csv")

    assert list(d3.columns) == ["user_id", "signup_date", "platform", "new_flow", "activated_7d"]
    assert len(d3) == 6000
    assert d3["user_id"].is_unique
    assert set(d3["new_flow"]) == {0, 1}
    assert set(d3["activated_7d"]) == {0, 1}
    assert abs(d3["new_flow"].mean() - 0.5) < 0.02  # 1:1 assignment


def test_d4_keeps_only_completers_from_the_same_roster() -> None:
    d3, d4 = read("d3_onboarding_ab.csv"), read("d4_onboarding_completers.csv")

    assert (d4["completed_onboarding"] == 1).all()
    assert set(d4["user_id"]) < set(d3["user_id"])
    assert len(d4) < 0.8 * len(d3)
    # the new flow lets more users finish, so the arms are no longer 1:1
    assert d4["new_flow"].mean() > 0.53


def test_staggered_panel_has_launches_at_different_months_and_never_treated_cities() -> None:
    panel = read("did_staggered_launch.csv")

    starts = panel[panel["launched"] == 1].groupby("city")["month"].min()
    assert starts.nunique() == 3
    assert panel["city"].nunique() - len(starts) == 5  # never launched
    assert len(panel) == panel["city"].nunique() * panel["month"].nunique()


def test_threshold_data_assigns_the_scholarship_at_score_60() -> None:
    rdd = read("rdd_scholarship.csv")

    assert (rdd["scholarship"] == (rdd["score"] >= 60).astype(int)).all()


def test_iv_data_omits_the_confounder() -> None:
    iv = read("iv_distance.csv")

    assert list(iv.columns) == ["person_id", "distance_km", "college", "wage"]
