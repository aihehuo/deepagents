"""Unit tests for the UCObserver class."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

from deepagents.observability import UCObserver


class MyTestObserver(UCObserver):
    """Subclass of UCObserver for testing."""

    uc_name: ClassVar[str] = "test_case"


def test_uc_observer_logging(tmp_path: Path) -> None:
    """Verify that UCObserver logs messages to the configured directory."""
    UCObserver._loggers.clear()
    # Set the log directory to the temporary path
    UCObserver.set_log_dir(tmp_path)

    # Log some messages
    MyTestObserver.info("Hello info")
    MyTestObserver.warn("Hello warning")
    MyTestObserver.error("Hello error")

    # Verify that the log file was created
    log_file = tmp_path / "uc_test_case.log"
    assert log_file.exists()

    # Verify content
    content = log_file.read_text(encoding="utf-8")
    assert "[INFO] Hello info" in content
    assert "[WARNING] Hello warning" in content
    assert "[ERROR] Hello error" in content


def test_uc_observer_newline_consolidation(tmp_path: Path) -> None:
    """Verify that UCObserver replaces newlines with spaces in log entries."""
    UCObserver._loggers.clear()
    UCObserver.set_log_dir(tmp_path)

    MyTestObserver.info("Line 1\nLine 2\r\nLine 3")

    log_file = tmp_path / "uc_test_case.log"
    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").splitlines()
    # Check that there is only one new log entry line
    last_line = lines[-1]
    assert "Line 1 Line 2 Line 3" in last_line
    assert "\n" not in last_line.replace("\r", "")[30:]  # Skip timestamp section


def test_uc_observer_failsafe() -> None:
    """Verify that UCObserver remains fail-safe even if log dir is invalid/unwritable."""
    UCObserver._loggers.clear()
    # Set log directory to an invalid path or a file that is not a directory
    invalid_path = "/nonexistent_directory/invalid_path/file.log"
    UCObserver.set_log_dir(invalid_path)

    # This should not raise any exception
    MyTestObserver.info("Should not crash")


def test_uc_observer_retry_after_failure(tmp_path: Path) -> None:
    """Verify that if initialization fails once, it is retried and succeeds when directory becomes valid."""
    from unittest.mock import patch

    UCObserver._loggers.clear()
    UCObserver.set_log_dir(tmp_path)

    # Initial attempt with mocked FileHandler failure should fail silently and not cache
    with patch("logging.FileHandler", side_effect=PermissionError("Mocked write failure")):
        logger_first = MyTestObserver.get_logger()
        assert len(logger_first.handlers) == 1
        assert isinstance(logger_first.handlers[0], logging.NullHandler)
        assert MyTestObserver.uc_name not in UCObserver._loggers

    # Next attempt without patch should succeed and cache the logger
    logger_second = MyTestObserver.get_logger()
    assert MyTestObserver.uc_name in UCObserver._loggers
    assert any(isinstance(h, logging.FileHandler) for h in logger_second.handlers)
