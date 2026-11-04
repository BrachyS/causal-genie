"""Turn a downloaded table into a recorded `Result`, so every reported number has an id.

The `Result` id is a hash of the inputs (data checksum, columns, cutoff, scale): the same data
and choices always give the same id, and any change gives a new one.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import pandas as pd

from causal_genie.agent.state import Result
from causal_genie.log import log_step
from causal_genie.tools.stats import rct_effect


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
        inputs_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        rec["output"] = outputs
        return Result(
            id=f"res-{inputs_hash[:8]}",
            tool="rct_effect",
            tool_version=version("causal-genie"),
            inputs_hash=inputs_hash,
            outputs=outputs,
        )
