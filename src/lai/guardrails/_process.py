"""Thin compatibility wrapper delegating subprocess isolation helpers to lai.observability."""

from __future__ import annotations

from lai.observability._process import (
    WindowsJobObject,
    attach_win32_job_object,
    close_win32_handle,
)
from lai.observability._process import (
    kill_asyncio_process_tree as kill_process_tree,
)
from lai.observability._process import (
    terminate_asyncio_process_tree as terminate_process_tree,
)

__all__ = [
    "WindowsJobObject",
    "attach_win32_job_object",
    "close_win32_handle",
    "kill_process_tree",
    "terminate_process_tree",
]
