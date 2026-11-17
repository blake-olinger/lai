"""Cross-platform subprocess isolation helpers (64-bit-safe Win32 Job Objects & POSIX groups)."""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import sys
from ctypes import wintypes
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import asyncio
    import subprocess
    from collections.abc import Callable

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100

__all__ = [
    "WindowsJobObject",
    "attach_win32_job_object",
    "close_win32_handle",
    "kill_asyncio_process_tree",
    "kill_popen_process_group",
    "terminate_asyncio_process_tree",
    "terminate_popen_process_group",
]


class _IoCounters(ctypes.Structure):
    """Win32 IO_COUNTERS structure layout."""

    _fields_ = (
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    )


class _JobObjectBasicLimitInfo(ctypes.Structure):
    """Win32 JOBOBJECT_BASIC_LIMIT_INFORMATION structure layout."""

    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    )


class _JobObjectExtendedLimitInfo(ctypes.Structure):
    """Win32 JOBOBJECT_EXTENDED_LIMIT_INFORMATION structure layout."""

    _fields_ = (
        ("BasicLimitInformation", _JobObjectBasicLimitInfo),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


class _ProcessHandle(Protocol):
    """Structural interface shared by asyncio.subprocess.Process and subprocess.Popen."""

    @property
    def pid(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


def _last_win32_error() -> int:
    """Return ctypes.get_last_error() safely when running on Windows."""
    get_last_error = cast("Callable[[], int] | None", getattr(ctypes, "get_last_error", None))
    return get_last_error() if get_last_error is not None else 0


def _load_kernel32() -> ctypes.CDLL | None:
    """Load kernel32 on Windows with explicit 64-bit-safe HANDLE signatures."""
    if sys.platform != "win32":
        return None
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return None
    k32 = cast("ctypes.CDLL", win_dll("kernel32", use_last_error=True))
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    return k32


class WindowsJobObject:
    """Win32 Job Object configured with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE."""

    def __init__(self) -> None:
        self._k32 = _load_kernel32()
        self._handle: int | None = None
        if self._k32 is None:
            return
        job_handle = self._k32.CreateJobObjectW(None, None)
        if not job_handle:
            raise OSError(_last_win32_error(), "CreateJobObjectW failed")
        info = _JobObjectExtendedLimitInfo()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = self._k32.SetInformationJobObject(
            job_handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            err = _last_win32_error()
            self._k32.CloseHandle(job_handle)
            raise OSError(err, "SetInformationJobObject failed")
        self._handle = int(job_handle)

    @property
    def active(self) -> bool:
        """Return whether an open Win32 Job Object handle is held."""
        return self._handle is not None

    @property
    def handle(self) -> int | None:
        """Return the raw Win32 Job Object handle if currently open."""
        return self._handle

    def assign_pid(self, pid: int) -> bool:
        """Attach an existing OS process by PID to this Job Object."""
        if self._k32 is None or self._handle is None or pid <= 0:
            return False
        proc_handle = self._k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not proc_handle:
            return False
        try:
            ok = self._k32.AssignProcessToJobObject(self._handle, proc_handle)
            return bool(ok)
        finally:
            self._k32.CloseHandle(proc_handle)

    def release_handle(self) -> int | None:
        """Detach and return the open Job Object handle without closing it."""
        handle = self._handle
        self._handle = None
        return handle

    def close(self) -> None:
        """Close the Job Object handle, terminating all assigned processes."""
        if self._k32 is not None and self._handle is not None:
            handle = self._handle
            self._handle = None
            self._k32.CloseHandle(handle)


def attach_win32_job_object(pid: int) -> int | None:
    """Attach pid to a Win32 Job Object configured with KILL_ON_JOB_CLOSE."""
    if pid <= 0 or sys.platform != "win32":
        return None
    try:
        job = WindowsJobObject()
    except OSError:
        return None
    if not job.active:
        return None
    if job.assign_pid(pid):
        return job.release_handle()
    job.close()
    return None


def close_win32_handle(handle: int | None) -> None:
    """Close a Win32 kernel handle safely if non-null."""
    if handle is None or handle <= 0 or sys.platform != "win32":
        return
    k32 = _load_kernel32()
    if k32 is not None:
        k32.CloseHandle(handle)


def _signal_process_tree(
    proc: _ProcessHandle,
    sig: int,
    fallback: Callable[[], None],
) -> None:
    """Signal a POSIX process group when available, falling back to direct process termination."""
    if sys.platform != "win32" and proc.pid > 0:
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
