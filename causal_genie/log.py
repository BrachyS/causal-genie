"""Logging so every run can be debugged from its log alone.

Each pipeline step logs one line with its inputs, outputs, tokens, latency and any error.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

step_logger = logging.getLogger("causal_genie.step")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger once, at program entry (CLI, UI, eval)."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


@contextmanager
def log_step(step: str, **inputs: Any) -> Generator[dict[str, Any]]:
    """Log one step's inputs, outputs, tokens and latency; on error, log it and re-raise.

    The caller fills the yielded dict (e.g. `rec["output"] = ...`, `rec["tokens"] = ...`);
    its keys are attached to the log record so tests and handlers can read them.
    """
    rec: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield rec
    except Exception as e:
        latency = time.perf_counter() - start
        step_logger.error(
            "step=%s inputs=%r error=%r latency_s=%.3f",
            step,
            inputs,
            e,
            latency,
            extra={"step": step, "inputs": inputs, "error": repr(e), "latency_s": latency, **rec},
        )
        raise
    latency = time.perf_counter() - start
    step_logger.info(
        "step=%s inputs=%r rec=%r latency_s=%.3f",
        step,
        inputs,
        rec,
        latency,
        extra={"step": step, "inputs": inputs, "latency_s": latency, **rec},
    )
