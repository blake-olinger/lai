"""Cross-platform subprocess isolation helpers (Win32 Job Object and POSIX groups)."""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import sys
from ctypes import wintypes
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import asyncio

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


class _JobObjectBasicLimitInfo(ctypes.Structure):
    """Win32 JOBOBJECT_BASIC_LIMIT_INFORMATION layout."""

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


class _IoCounters(ctypes.Structure):
    """Win32 IO_COUNTERS layout."""

    _fields_ = (
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    )


class _JobObjectExtendedLimitInfo(ctypes.Structure):
    """Win32 JOBOBJECT_EXTENDED_LIMIT_INFORMATION layout."""

    _fields_ = (
        ("BasicLimitInformation", _JobObjectBasicLimitInfo),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


def _assign_pid_to_job(job_handle: int, pid: int) -> bool:
    """Open process handle for pid and bind it to an existing Win32 Job Object."""
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return False
    kernel32 = windll.kernel32
    proc_handle = int(kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid))
    if proc_handle == 0:
        return False
    try:
        return bool(
            kernel32.AssignProcessToJobObject(
                wintypes.HANDLE(job_handle),
                wintypes.HANDLE(proc_handle),
            ),
        )
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(proc_handle))


def attach_win32_job_object(pid: int) -> int | None:
    """Attach pid to a Win32 Job Object configured with KILL_ON_JOB_CLOSE."""
    if sys.platform != "win32":
        return None
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    kernel32 = windll.kernel32
    job_handle = int(kernel32.CreateJobObjectW(None, None))
    if job_handle == 0:
        return None

    info = _JobObjectExtendedLimitInfo()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok_info = bool(
        kernel32.SetInformationJobObject(
            wintypes.HANDLE(job_handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ),
    )
    if ok_info and _assign_pid_to_job(job_handle, pid):
        return job_handle
    kernel32.CloseHandle(wintypes.HANDLE(job_handle))
    return None


def close_win32_handle(handle: int | None) -> None:
    """Close a Win32 kernel handle safely if non-null."""
    if handle is None or sys.platform != "win32":
        return
    windll = getattr(ctypes, "windll", None)
    if windll is not None:
        windll.kernel32.CloseHandle(wintypes.HANDLE(handle))


def terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Send graceful termination signal to the supervised process or POSIX group."""
    if sys.platform != "win32":
        killpg = getattr(os, "killpg", None)
        getpgid = getattr(os, "getpgid", None)
        if callable(killpg) and callable(getpgid):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                pgid = cast("int", getpgid(proc.pid))
                killpg(pgid, signal.SIGTERM)
                return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()


def kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Forcefully terminate the supervised process or POSIX group after timeout."""
    if sys.platform != "win32":
        killpg = getattr(os, "killpg", None)
        getpgid = getattr(os, "getpgid", None)
        sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
        if callable(killpg) and callable(getpgid):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                pgid = cast("int", getpgid(proc.pid))
                killpg(pgid, sigkill)
                return
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
