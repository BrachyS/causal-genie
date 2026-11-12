"""`detect_design` on fixtures with known designs (T9.4, T9.4.1)."""

import numpy as np
import pandas as pd
from conftest import FIXTURES

from causal_genie.agent.state import DagEdge
from causal_genie.tools.identify import detect_design


def edge(source: str, target: str, cited: bool = True) -> DagEdge:
    return DagEdge(source=source, target=target, claim_ids=["c1"] if cited else [])


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name)


AB_DAG = [edge("new_flow", "activated_7d"), edge("platform", "activated_7d")]


def test_d3_documented_ab_test_is_randomized() -> None:
    d = detect_design(read("d3_onboarding_ab.csv"), "new_flow", "activated_7d", AB_DAG, True)

    assert (d.design, d.estimable, d.estimand) == ("randomized", True, "ATE")


def test_d4_completers_only_export_is_selection_after_assignment() -> None:
    d = detect_design(
        read("d4_onboarding_completers.csv"), "new_flow", "activated_7d", AB_DAG, True
    )

    assert (d.design, d.estimable) == ("selection_after_assignment", False)
    assert "roster" in d.check
    assert any("arm sizes" in r for r in d.reasons)


def test_d1_one_missing_patient_is_not_selection() -> None:
    # 235 rows of 236 randomized, 117 vs 118 per arm
    table = pd.DataFrame({"treat": [1] * 117 + [0] * 118, "sore": [0, 1] * 117 + [0]})

    d = detect_design(table, "treat", "sore", [edge("treat", "sore")], True, roster=236)

    assert d.design == "randomized"


def test_rows_well_short_of_the_roster_are_selection_even_with_balanced_arms() -> None:
    table = pd.DataFrame({"treat": [1, 0] * 100, "y": [0, 1] * 100})

    d = detect_design(table, "treat", "y", [edge("treat", "y")], True, roster=300)

    assert d.design == "selection_after_assignment"
    assert any("200 rows" in r and "300" in r for r in d.reasons)


def test_undocumented_assignment_is_never_called_randomized() -> None:
    d = detect_design(read("d3_onboarding_ab.csv"), "new_flow", "activated_7d", AB_DAG, False)

    assert d.design == "observational"


def test_threshold_assigned_data_is_rdd_at_the_cutoff() -> None:
    dag = [edge("score", "scholarship"), edge("score", "gpa"), edge("scholarship", "gpa")]

    d = detect_design(read("rdd_scholarship.csv"), "scholarship", "gpa", dag, False)

    assert (d.design, d.estimable, d.estimand) == ("RDD", False, "effect at the cutoff")
    assert any("score" in r and "60" in r for r in d.reasons)


def test_staggered_launch_panel_is_did() -> None:
    dag = [edge("launched", "orders")]

    d = detect_design(
        read("did_staggered_launch.csv"),
        "launched",
        "orders",
        dag,
        False,
        unit="city",
        time="month",
    )

    assert (d.design, d.estimable, d.estimand) == ("DiD", False, "ATT")
    assert "parallel trends" in d.key_assumption


IV_DAG = [
    edge("distance_km", "college"),
    edge("ability", "college"),
    edge("ability", "wage"),
    edge("college", "wage"),
]


def test_cited_instrument_with_unobserved_confounder_is_iv() -> None:
    d = detect_design(read("iv_distance.csv"), "college", "wage", IV_DAG, False)

    assert (d.design, d.estimable, d.estimand) == ("IV", False, "LATE")
    assert any("distance_km" in r for r in d.reasons)


def test_uncited_instrument_is_not_iv_and_names_the_missing_confounder() -> None:
    dag = [edge("distance_km", "college", cited=False), *IV_DAG[1:]]

    d = detect_design(read("iv_distance.csv"), "college", "wage", dag, False)

    assert (d.design, d.estimable) == ("observational", False)
    assert "ability" in d.check


def test_observed_confounder_is_estimable_observational() -> None:
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"x": rng.integers(0, 2, 100), "y": rng.normal(size=100),
                          "c": rng.normal(size=100)})  # fmt: skip
    dag = [edge("c", "x"), edge("c", "y"), edge("x", "y")]

    d = detect_design(table, "x", "y", dag, False)

    assert (d.design, d.estimable, d.estimand) == ("observational", True, "ATE")
    assert "c" in d.key_assumption
