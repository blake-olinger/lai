"""Thin compatibility wrapper delegating POSIX process tree helpers to lai.observability."""

from __future__ import annotations

from lai.observability._process import (
    kill_asyncio_process_tree as kill_process_tree,
)
from lai.observability._process import (
    make_pdeathsig_preexec,
)
from lai.observability._process import (
    terminate_asyncio_process_tree as terminate_process_tree,
)

__all__ = [
    "kill_process_tree",
    "make_pdeathsig_preexec",
    "terminate_process_tree",
]
