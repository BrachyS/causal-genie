"""Turn a downloaded table into a recorded `Result`, so every reported number has an id.

The `Result` id is a hash of the inputs (data checksum, columns, cutoff or adjustment set,
scale): the same data and choices always give the same id, and any change gives a new one.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import pandas as pd

from causal_genie.agent.state import Result
from causal_genie.log import log_step
from causal_genie.tools.stats import backdoor_effect, rct_effect


def estimate_rct(
    path: Path,
    data_sha256: str,
    treatment: str,
    outcome: str,
    cutoff: float | None,
    scale: Literal["risk_difference", "mean_difference"],
) -> Result:
    """Estimate a randomized effect from a CSV, analyzed as assigned.

    `cutoff` makes a binary outcome (`outcome > cutoff`) from a score; rows missing treatment or
    outcome are excluded and counted, never silently dropped.
    """
    inputs = {
        "data_sha256": data_sha256,
        "treatment": treatment,
        "outcome": outcome,
        "cutoff": cutoff,
        "scale": scale,
    }
    with log_step("estimate", **inputs) as rec:
        table = pd.read_csv(path)
        data = table[[treatment, outcome]].dropna()
        if cutoff is not None:
            data[outcome] = (data[outcome] > cutoff).astype(int)
        outputs = rct_effect(data, treatment, outcome, scale) | {
            "rows": len(table),
            "rows_excluded_missing": len(table) - len(data),
            "treatment_column": treatment,
            "outcome_column": outcome,
            "cutoff": cutoff,
        }
        rec["output"] = outputs
        return result("rct_effect", inputs, outputs)


def result(tool: str, inputs: dict, outputs: dict) -> Result:
    """A `Result` whose id is a hash of the inputs, so the same choices give the same id."""
    inputs_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return Result(
        id=f"res-{inputs_hash[:8]}",
        tool=tool,
        tool_version=version("causal-genie"),
        inputs_hash=inputs_hash,
        outputs=outputs,
    )


def estimate_backdoor(
    table: pd.DataFrame,
    data_sha256: str,
    exposure: str,
    outcome: str,
    adjust_for: list[str],
    scale: Literal["risk_difference", "mean_difference", "odds_ratio"],
) -> Result:
    """Estimate an observational effect from the analysis table (one column per node),
    adjusting for the backdoor set from `identify`.

    Rows missing any used column are excluded and counted, never silently dropped.
    """
    inputs = {
        "data_sha256": data_sha256,
        "exposure": exposure,
        "outcome": outcome,
        "adjust_for": adjust_for,
        "scale": scale,
    }
    with log_step("estimate", **inputs) as rec:
        data = table[[exposure, outcome, *adjust_for]].dropna()
        outputs = backdoor_effect(data, exposure, outcome, adjust_for, scale) | {
            "rows": len(table),
            "rows_excluded_missing": len(table) - len(data),
        }
        rec["output"] = outputs
        return result("backdoor_effect", inputs, outputs)
