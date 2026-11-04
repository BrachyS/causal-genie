from pathlib import Path

import pytest

from causal_genie.agent.steps.estimate import estimate_rct


@pytest.fixture
def trial_csv(tmp_path: Path) -> Path:
    """Tiny trial: 4 treated (one missing outcome) and 4 controls."""
    path = tmp_path / "trial.csv"
    path.write_text("treat,pain\n1,0\n1,2\n1,0\n1,\n0,3\n0,1\n0,0\n0,5\n")
    return path


def test_binary_outcome_from_cutoff_with_missing_rows_excluded(trial_csv: Path) -> None:
    r = estimate_rct(
        trial_csv, "sha", treatment="treat", outcome="pain", cutoff=0, scale="risk_difference"
    )

    out = r.outputs
    assert (out["events_treated"], out["n_treated"]) == (1, 3)  # pain > 0: one of 0, 2, 0
    assert (out["events_control"], out["n_control"]) == (3, 4)
    assert (out["rows"], out["rows_excluded_missing"]) == (8, 1)
    assert (out["treatment_column"], out["outcome_column"], out["cutoff"]) == ("treat", "pain", 0)
    assert r.tool == "rct_effect"


def test_same_inputs_give_same_result_id(trial_csv: Path) -> None:
    def run(cutoff: float) -> str:
        return estimate_rct(trial_csv, "sha", "treat", "pain", cutoff, "risk_difference").id

    assert run(0) == run(0)
    assert run(0) != run(1)
