import logging

import pytest


@pytest.fixture
def step_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture causal_genie step logs at INFO and above."""
    caplog.set_level(logging.INFO, logger="causal_genie")
    return caplog
