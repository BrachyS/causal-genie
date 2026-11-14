import numpy as np
import pandas as pd
import pytest

from causal_genie.agent.state import DagEdge, DeriveRule, Mapping, NodeMapping
from causal_genie.agent.steps.estimate import estimate_backdoor, estimate_rct
from causal_genie.tools.derive import node_table, us_holiday
from causal_genie.tools.identify import identify, precision_set

MAPPED_FROM = {"treated": "treat", "sore": "pain > 0"}


@pytest.fixture
def trial() -> pd.DataFrame:
    """Tiny trial as a node table (cutoff already applied): 4 treated, one missing outcome."""
    return pd.DataFrame({"treated": [1, 1, 1, 1, 0, 0, 0, 0], "sore": [0, 1, 0, None, 1, 1, 0, 1]})


def test_rct_with_missing_rows_excluded_and_columns_recorded(trial: pd.DataFrame) -> None:
    r = estimate_rct(trial, "sha", "treated", "sore", "risk_difference", MAPPED_FROM)

    out = r.outputs
    assert (out["events_treated"], out["n_treated"]) == (1, 3)
    assert (out["events_control"], out["n_control"]) == (3, 4)
    assert (out["rows"], out["rows_excluded_missing"]) == (8, 1)
    assert out["mapped_from"] == MAPPED_FROM
    assert r.tool == "rct_effect"


def test_same_inputs_give_same_result_id(trial: pd.DataFrame) -> None:
    def run(mapped_from: dict[str, str]) -> str:
        return estimate_rct(trial, "sha", "treated", "sore", "risk_difference", mapped_from).id

    assert run(MAPPED_FROM) == run(MAPPED_FROM)
    assert run(MAPPED_FROM) != run(MAPPED_FROM | {"sore": "pain > 1"})


def test_backdoor_result_counts_excluded_rows_and_records_the_set() -> None:
    table = pd.DataFrame(
        {
            "x": [1, 1, 0, 0, 1, 0, 1, 0],
            "y": [3.0, 4, 1, 2, 5, 1, None, 2],
            "c": [1, 2, 1, 2, 3, 0, 1, 1],
        }  # fmt: skip
    )

    r = estimate_backdoor(
        table, "sha", "x", "y", ["c"], "mean_difference", precision=[], mapped_from={"x": "x"}
    )

    assert r.tool == "backdoor_effect"
    assert (r.outputs["rows"], r.outputs["rows_excluded_missing"], r.outputs["n"]) == (8, 1, 7)
    assert (r.outputs["adjusted_for"], r.outputs["precision"]) == (["c"], [])
    assert r.outputs["mapped_from"] == {"x": "x"}
    unadjusted = estimate_backdoor(table, "sha", "x", "y", [], "mean_difference", [], {"x": "x"})
    assert r.id != unadjusted.id


def test_precision_variables_are_adjusted_for_and_recorded_apart() -> None:
    table = pd.DataFrame(
        {
            "x": [1, 0, 1, 0, 1, 0, 1, 0],
            "y": [5.0, 1, 6, 2, 4, 1, 7, 3],
            "p": [1, 0, 2, 1, 0, 0, 3, 2],
        }
    )

    r = estimate_backdoor(table, "sha", "x", "y", [], "mean_difference", ["p"], {})

    assert (r.outputs["adjusted_for"], r.outputs["precision"]) == ([], ["p"])
    assert r.outputs["estimate"] != r.outputs["naive_estimate"]  # p was in the model


def node(name: str, columns: list[str], derive: DeriveRule | None = None) -> NodeMapping:
    return NodeMapping(node=name, columns=columns, derive=derive, cutoff=None, categorical=False)


def test_d0_calendar_causes_of_births_in_the_dag_are_adjusted_for_precision() -> None:
    """T10.9 / option (a): once the DAG holds the outcome's other causes, the D0 pipeline
    (mapping → identify → precision → estimate) adjusts for them and the CI narrows a lot.

    Edges are those of live run 20261114-154259-6bb1d2 plus the calendar nodes; the table is
    a synthetic copy of the D0 births table with calendar effects and no moon effect.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("1994-01-01", "2003-12-31", freq="D")
    raw = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "year": dates.year})
    weekend = (dates.day_of_week >= 5).astype(int)
    births = (
        12000
        - 3000 * weekend
        - 2500 * us_holiday(pd.Series(dates)).to_numpy()
        + 400 * np.isin(dates.month, [7, 8, 9])
        + 50 * (dates.year - 1994)
        + rng.normal(0, 150, len(dates))
    )
    raw["births"] = births.round()
    edges = [
        DagEdge(source="full_moon", target="births", claim_ids=[], question=True),
        DagEdge(source="full_moon", target="moonlight", claim_ids=["c1"]),
        DagEdge(source="moonlight", target="melatonin", claim_ids=[]),
        DagEdge(source="melatonin", target="births", claim_ids=["c2"]),
        *(
            DagEdge(source=n, target="births", claim_ids=[])
            for n in ("day_of_week", "holiday", "season", "year")
        ),
    ]
    mapping = Mapping(
        nodes=[
            node("full_moon", ["date"], "full_moon"),
            node("births", ["births"]),
            node("moonlight", []),
            node("melatonin", []),
            node("day_of_week", ["date"], "day_of_week"),
            node("holiday", ["date"], "holiday"),
            node("season", ["date"], "season"),
            node("year", ["year"]),
        ],
        unit=None,
        time=None,
    )
    table = node_table(raw, mapping)
    observed = list(table.columns)

    adjustment = identify(edges, "full_moon", "births", observed, randomized=False)
    precision = precision_set(edges, "full_moon", "births", [], observed)
    out = estimate_backdoor(
        table, "sha", "full_moon", "births", [], "mean_difference", precision, {}
    ).outputs

    assert adjustment.adjust_for == []
    assert precision == ["day_of_week", "holiday", "season", "year"]
    width, naive_width = out["ci_high"] - out["ci_low"], out["naive_ci_high"] - out["naive_ci_low"]
    assert width < naive_width / 5
    assert out["ci_low"] < 0 < out["ci_high"]  # no moon effect in the data
