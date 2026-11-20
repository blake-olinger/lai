"""POSIX subprocess isolation, process-group signaling, and Linux PR_SET_PDEATHSIG helpers."""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import sys
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import asyncio
    import subprocess
    from collections.abc import Callable

_PR_SET_PDEATHSIG = 1

__all__ = [
    "kill_asyncio_process_tree",
    "kill_popen_process_group",
    "make_pdeathsig_preexec",
    "terminate_asyncio_process_tree",
    "terminate_popen_process_group",
]


class _ProcessHandle(Protocol):
    """Structural interface shared by asyncio.subprocess.Process and subprocess.Popen."""

    @property
    def pid(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class _PrctlFunc(Protocol):
    """Typed view over ctypes.CDLL.prctl."""

    argtypes: list[object]
    restype: object

    def __call__(self, option: int, arg2: int) -> int: ...


def _load_libc_prctl() -> Callable[[int, int], int] | None:
    """Load Linux libc.prctl when running on Linux; return None elsewhere."""
    if not sys.platform.startswith("linux"):
        return None
    libc: ctypes.CDLL | None = None
    for lib_name in ("libc.so.6", None):
        try:
            libc = ctypes.CDLL(lib_name)
            break
        except OSError:
            continue
    if libc is None:
        return None
    prctl: object = getattr(libc, "prctl", None)
    if not callable(prctl):
        return None
    if hasattr(prctl, "argtypes"):
        c_prctl = cast("_PrctlFunc", prctl)
        c_prctl.argtypes = [ctypes.c_int, ctypes.c_ulong]
        c_prctl.restype = ctypes.c_int
    return cast("Callable[[int, int], int]", prctl)


def make_pdeathsig_preexec(sig: int = signal.SIGTERM) -> Callable[[], None]:
    """Return a POSIX preexec_fn requesting Linux PR_SET_PDEATHSIG (no-op off Linux)."""
    prctl = _load_libc_prctl()

    def _set_pdeathsig() -> None:
        if prctl is not None:
            with contextlib.suppress(OSError, ValueError):
                prctl(_PR_SET_PDEATHSIG, sig)

    return _set_pdeathsig


def _signal_process_tree(
    proc: _ProcessHandle,
    sig: int,
    fallback: Callable[[], None],
) -> None:
    """Signal a POSIX process group when available, falling back to direct process signaling."""
    if proc.pid > 0:
        killpg = getattr(os, "killpg", None)
        getpgid = getattr(os, "getpgid", None)
        if callable(killpg) and callable(getpgid):
            with contextlib.suppress(ProcessLookupError, OSError):
                pgid = cast("int", getpgid(proc.pid))
                if pgid > 0:
                    killpg(pgid, sig)
                    return
    with contextlib.suppress(ProcessLookupError, OSError):
        fallback()


def terminate_asyncio_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Send graceful termination signal to an asyncio subprocess or its POSIX group."""
    _signal_process_tree(proc, signal.SIGTERM, proc.terminate)


def kill_asyncio_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Forcefully terminate an asyncio subprocess or its POSIX group."""
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    _signal_process_tree(proc, sigkill, proc.kill)


def terminate_popen_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Send graceful termination signal to a Popen subprocess or its POSIX group."""
    _signal_process_tree(proc, signal.SIGTERM, proc.terminate)


def kill_popen_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Forcefully terminate a Popen subprocess or its POSIX group."""
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    _signal_process_tree(proc, sigkill, proc.kill)
