"""Observability module providing per-use-case (UC) logging.

Each use case gets its own dedicated log file (uc_<name>.log) and its own
Logger instance at INFO level, fully independent of the system-wide
application logger level. Every public logging method is fail-safe.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar


class UCObserver:
    """Base class for Use Case (UC) observability logging in Python.

    Subclasses must override `uc_name` (e.g. "16_ai_cofounder") to specify
    their dedicated log filename: uc_<uc_name>.log.
    """

    uc_name: ClassVar[str] = ""
    _log_dir: ClassVar[Path | None] = None
    _loggers: ClassVar[dict[str, logging.Logger]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def set_log_dir(cls, directory: Path | str) -> None:
        """Set the base directory where UC logs will be stored.

        Args:
            directory: Path to the log directory
        """
        with cls._lock:
            cls._log_dir = Path(directory)
            with contextlib.suppress(OSError):
                cls._log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get or initialize a dedicated Logger for the subclass.

        Returns:
            Configured logging.Logger instance

        Raises:
            NotImplementedError: If uc_name is not defined on the subclass
        """
        if not cls.uc_name:
            msg = "Subclasses of UCObserver must define a non-empty `uc_name`."
            raise NotImplementedError(msg)

        logger_key = cls.uc_name

        # Double-checked locking pattern for thread safety
        if logger_key in cls._loggers:
            return cls._loggers[logger_key]

        with cls._lock:
            if logger_key in cls._loggers:
                return cls._loggers[logger_key]

            # Determine log directory path
            log_dir = cls._log_dir
            if log_dir is None:
                log_dir = Path.home() / ".deepagents" / "logs"

            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"uc_{cls.uc_name}.log"
            except Exception:  # noqa: BLE001  # fail-safe fallback to /tmp
                # In case home directory is unwritable, fallback to /tmp or current folder
                log_path = Path("/tmp") / f"uc_{cls.uc_name}.log"  # noqa: S108  # safe fallback

            logger = logging.getLogger(f"uc_observer.{cls.uc_name}")
            logger.setLevel(logging.INFO)
            logger.propagate = False  # Avoid propagating to root/uvicorn logger

            # Clear any existing handlers
            logger.handlers.clear()

            try:
                handler = logging.FileHandler(log_path, encoding="utf-8")
                handler.setLevel(logging.INFO)

                # Custom formatter to match the requested format:
                # [2026-07-10 14:35:20.123] [INFO] message
                class UCFormatter(logging.Formatter):
                    def format(self, record: logging.LogRecord) -> str:
                        dt = datetime.fromtimestamp(record.created, tz=UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        return f"[{dt}] [{record.levelname}] {record.getMessage()}"

                handler.setFormatter(UCFormatter())
                logger.addHandler(handler)
                cls._loggers[logger_key] = logger
            except Exception:  # noqa: BLE001  # fail-safe fallback to NullHandler
                # If file handler creation fails, add a NullHandler to prevent traceback logging
                logger.addHandler(logging.NullHandler())

            return logger

    @classmethod
    def _emit(cls, level: int, msg: str) -> None:
        """Fail-safe logger emit wrapper. Never raises exceptions.

        Args:
            level: Logging level (e.g. logging.INFO)
            msg: Log message string
        """
        try:
            logger = cls.get_logger()
            # Collapse all CR/LF into a single space to maintain "one event = one line"
            sanitized_msg = re.sub(r"[\r\n]+", " ", str(msg))
            logger.log(level, sanitized_msg)
        except Exception as e:  # noqa: BLE001  # fail-safe emit wrapper
            # Last-resort fallback to standard logging; fail-safe for target execution path
            try:
                standard_logger = logging.getLogger("uvicorn.error")
                standard_logger.warning(
                    "[UCObserver:%s] observe failed: %s: %s",
                    cls.uc_name,
                    type(e).__name__,
                    str(e),
                )
            except Exception:  # noqa: BLE001, S110  # final silent fallback
                pass

    @classmethod
    def info(cls, msg: str) -> None:
        """Log info level message.

        Args:
            msg: Log message
        """
        cls._emit(logging.INFO, msg)

    @classmethod
    def warn(cls, msg: str) -> None:
        """Log warning level message.

        Args:
            msg: Log message
        """
        cls._emit(logging.WARNING, msg)

    @classmethod
    def error(cls, msg: str) -> None:
        """Log error level message.

        Args:
            msg: Log message
        """
        cls._emit(logging.ERROR, msg)
