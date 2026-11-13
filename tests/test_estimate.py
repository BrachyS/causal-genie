import pandas as pd
import pytest

from causal_genie.agent.steps.estimate import estimate_backdoor, estimate_rct

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
