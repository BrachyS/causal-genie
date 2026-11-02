import pytest

from causal_genie.log import log_step


def step_records(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [r.__dict__ for r in caplog.records if r.name == "causal_genie.step"]


def test_log_step_records_inputs_outputs_latency(step_logs: pytest.LogCaptureFixture) -> None:
    with log_step("frame", question="Does the full moon cause more births?") as rec:
        rec["output"] = {"exposure": "full moon"}
        rec["tokens"] = 42

    (record,) = step_records(step_logs)
    assert record["levelname"] == "INFO"
    assert record["step"] == "frame"
    assert record["inputs"] == {"question": "Does the full moon cause more births?"}
    assert record["output"] == {"exposure": "full moon"}
    assert record["tokens"] == 42
    assert record["latency_s"] >= 0


def test_log_step_logs_error_and_reraises(step_logs: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ValueError, match="bad schema"), log_step("extract", url="https://x"):
        raise ValueError("bad schema")

    (record,) = step_records(step_logs)
    assert record["levelname"] == "ERROR"
    assert record["step"] == "extract"
    assert "bad schema" in record["error"]
    assert record["latency_s"] >= 0
