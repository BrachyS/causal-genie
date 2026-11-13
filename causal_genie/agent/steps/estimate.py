"""Turn a downloaded table into a recorded `Result`, so every reported number has an id.

The `Result` id is a hash of the inputs (data checksum, columns, cutoff or adjustment set,
scale): the same data and choices always give the same id, and any change gives a new one.
"""

import hashlib
import json
from importlib.metadata import version
from typing import Literal

import pandas as pd

from causal_genie.agent.state import Result
from causal_genie.log import log_step
from causal_genie.tools.stats import backdoor_effect, rct_effect


def estimate_rct(
    table: pd.DataFrame,
    data_sha256: str,
    exposure: str,
    outcome: str,
    scale: Literal["risk_difference", "mean_difference"],
    mapped_from: dict[str, str],
) -> Result:
    """Estimate a randomized effect from the analysis table (one column per node, cutoffs
    already applied), analyzed as assigned.

    `mapped_from` says which columns each node came from, for the report. Rows missing the
    exposure or outcome are excluded and counted, never silently dropped.
    """
    inputs = {
        "data_sha256": data_sha256,
        "exposure": exposure,
        "outcome": outcome,
        "scale": scale,
        "mapped_from": mapped_from,
    }
    with log_step("estimate", **inputs) as rec:
        data = table[[exposure, outcome]].dropna()
        outputs = rct_effect(data, exposure, outcome, scale) | {
            "rows": len(table),
            "rows_excluded_missing": len(table) - len(data),
            "mapped_from": mapped_from,
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
    precision: list[str],
    mapped_from: dict[str, str],
) -> Result:
    """Estimate an observational effect from the analysis table (one column per node),
    adjusting for the backdoor set from `identify` plus the `precision` variables (causes of
    the outcome only), which are recorded apart so the report does not call them confounders.

    Rows missing any used column are excluded and counted, never silently dropped.
    """
    inputs = {
        "data_sha256": data_sha256,
        "exposure": exposure,
        "outcome": outcome,
        "adjust_for": adjust_for,
        "precision": precision,
        "scale": scale,
        "mapped_from": mapped_from,
    }
    with log_step("estimate", **inputs) as rec:
        covariates = [*adjust_for, *precision]
        data = table[[exposure, outcome, *covariates]].dropna()
        outputs = backdoor_effect(data, exposure, outcome, covariates, scale) | {
            "adjusted_for": adjust_for,
            "precision": precision,
            "rows": len(table),
            "rows_excluded_missing": len(table) - len(data),
            "mapped_from": mapped_from,
        }
        rec["output"] = outputs
        return result("backdoor_effect", inputs, outputs)
