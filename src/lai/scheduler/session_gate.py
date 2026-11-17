"""Per-session mutual exclusion gate preserving arrival ordering."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lai.domain.errors import ValidationFailedError
from lai.ports.scheduler import SessionGatePort

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@dataclass(slots=True)
class _SessionLockEntry:
    """Reference-counted per-session asyncio lock."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ref_count: int = 0


class SessionGate(SessionGatePort):
    """Per-session mutual exclusion gate pruning idle locks on release."""

    def __init__(self) -> None:
        self._entries: dict[str, _SessionLockEntry] = {}

    @property
    def tracked_session_count(self) -> int:
        """Return the number of sessions with active or waiting lock entries."""
        return len(self._entries)

    def is_locked(self, session_id: str) -> bool:
        """Return whether session_id currently holds an active lock."""
        entry = self._entries.get(session_id)
        return entry is not None and entry.lock.locked()

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncGenerator[None]:
        """Acquire exclusive access for turns targeting session_id in arrival order."""
        if not session_id:
            msg = "session_id must not be empty."
            raise ValidationFailedError(msg)

        entry = self._entries.get(session_id)
        if entry is None:
            entry = _SessionLockEntry()
            self._entries[session_id] = entry

        entry.ref_count += 1
        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            entry.ref_count -= 1
            if entry.ref_count == 0 and self._entries.get(session_id) is entry:
                del self._entries[session_id]
