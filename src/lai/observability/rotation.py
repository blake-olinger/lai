"""24-hour UTC rotating file handlers and thread-safe stream writers."""

from __future__ import annotations

import gzip
import logging
import logging.handlers
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType


def _gzip_namer(default_name: str) -> str:
    """Append .gz extension to rotated backup filenames."""
    return f"{default_name}.gz"


def _gzip_rotator(source: str, dest: str) -> None:
    """Compress source log file into dest atomically and remove source."""
    source_path = Path(source)
    if not source_path.exists():
        return
    dest_path = Path(dest)
    tmp_path = dest_path.with_name(f"{dest_path.name}.tmp")
    try:
        with source_path.open("rb") as src_file, gzip.open(tmp_path, "wb") as gz_file:
            shutil.copyfileobj(src_file, gz_file)
        tmp_path.replace(dest_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise
    source_path.unlink(missing_ok=True)


class SafeTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """TimedRotatingFileHandler resilient to transient Windows file locks."""

    def doRollover(self) -> None:  # noqa: N802
        """Perform rollover while recovering gracefully from transient PermissionError."""
        try:
            super().doRollover()
        except PermissionError:
            current_time = int(time.time())
            self.rolloverAt = self.computeRollover(current_time)
            if self.stream is not None:  # pyright: ignore[reportUnnecessaryComparison]
                self.stream.close()
                self.stream = None  # pyright: ignore[reportAttributeAccessIssue]


def create_daily_rotating_handler(  # noqa: PLR0913, PLR0917
    log_path: Path,
    level: int,
    retention_days: int = 14,
    compress: bool = True,
    when: str = "midnight",
    interval: int = 1,
) -> logging.handlers.TimedRotatingFileHandler:
    """Create a UTC daily rotating handler with optional gzip compression and pruning."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = SafeTimedRotatingFileHandler(
        filename=str(log_path),
        when=when,
        interval=interval,
        backupCount=retention_days,
        encoding="utf-8",
        utc=True,
    )
    handler.setLevel(level)
    if compress:
        handler.namer = _gzip_namer
        handler.rotator = _gzip_rotator
    return handler


class DailyRotatingStreamWriter:
    """Thread-safe line writer backed by a 24-hour rotating file handler."""

    def __init__(  # noqa: PLR0913
        self,
        log_path: Path,
        *,
        retention_days: int = 14,
        compress: bool = True,
        when: str = "midnight",
        interval: int = 1,
        logger_name: str = "lai.stream",
    ) -> None:
        self._log_path = log_path
        self._logger_name = logger_name
        self._lock = threading.Lock()
        self._closed = False
        self._handler = create_daily_rotating_handler(
            log_path=log_path,
            level=logging.INFO,
            retention_days=retention_days,
            compress=compress,
            when=when,
            interval=interval,
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))

    @property
    def handler(self) -> logging.handlers.TimedRotatingFileHandler:
        """Return the underlying TimedRotatingFileHandler."""
        return self._handler

    def write_line(self, line: str) -> None:
        """Write a single line cleanly through the rotating handler."""
        clean_line = line.rstrip("\r\n")
        record = logging.LogRecord(
            name=self._logger_name,
            level=logging.INFO,
            pathname=str(self._log_path),
            lineno=0,
            msg=clean_line,
            args=(),
            exc_info=None,
        )
        with self._lock:
            if self._closed:
                return
            self._handler.emit(record)
            self._handler.flush()

    def flush(self) -> None:
        """Flush buffered output to disk."""
        with self._lock:
            if not self._closed:
                self._handler.flush()

    def close(self) -> None:
        """Close the underlying rotating file handler idempotently."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._handler.flush()
            self._handler.close()

    def __enter__(self) -> DailyRotatingStreamWriter:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the writer when leaving context."""
        del exc_type, exc, tb
        self.close()
