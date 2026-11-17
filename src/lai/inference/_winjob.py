"""OS process-group and Windows Job Object lifecycle helpers."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.wintypes as wt
import os
import signal
import sys
from typing import IO, TYPE_CHECKING

from pydantic import Field, ValidationError

from lai.domain.base import StrictModel

if TYPE_CHECKING:
    import subprocess

    from lai.observability.rotation import DailyRotatingStreamWriter

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_MIN_GPU_CSV_FIELDS = 4
_GPU_TEMP_FIELD_INDEX = 4
_MAX_UTIL_PERCENT = 100


class GpuTelemetry(StrictModel):
    """Snapshot of NVIDIA GPU memory and utilization reported by nvidia-smi."""

    gpu_name: str = Field(min_length=1)
    memory_used_mb: int = Field(ge=0)
    memory_total_mb: int = Field(ge=0)
    utilization_percent: int = Field(ge=0, le=_MAX_UTIL_PERCENT)
    temperature_c: float | None = None


def _parse_optional_float(token: str) -> float | None:
    cleaned = token.strip()
    if not cleaned or cleaned.startswith("["):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_nvidia_smi_output(stdout: str) -> GpuTelemetry | None:
    """Parse CSV output from nvidia-smi into GpuTelemetry."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < _MIN_GPU_CSV_FIELDS:
        return None
    try:
        temp = (
            _parse_optional_float(parts[_GPU_TEMP_FIELD_INDEX])
            if len(parts) > _GPU_TEMP_FIELD_INDEX
            else None
        )
        return GpuTelemetry(
            gpu_name=parts[0],
            memory_used_mb=int(float(parts[1])),
            memory_total_mb=int(float(parts[2])),
            utilization_percent=int(float(parts[3])),
            temperature_c=temp,
        )
    except (ValueError, ValidationError):
        return None


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wt.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wt.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wt.DWORD),
        ("SchedulingClass", wt.DWORD),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _load_kernel32() -> ctypes.CDLL | None:
    """Load kernel32 on Windows with strict 64-bit HANDLE signatures."""
    if sys.platform != "win32":
        return None
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return None
    k32: ctypes.CDLL = win_dll("kernel32", use_last_error=True)
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.SetInformationJobObject.argtypes = [
        wt.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wt.DWORD,
    ]
    k32.SetInformationJobObject.restype = wt.BOOL
    k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    k32.OpenProcess.restype = wt.HANDLE
    k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    k32.AssignProcessToJobObject.restype = wt.BOOL
    k32.CloseHandle.argtypes = [wt.HANDLE]
    k32.CloseHandle.restype = wt.BOOL
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
            err = ctypes.get_last_error()
            raise OSError(err, "CreateJobObjectW failed")
        info = _JobObjectExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = self._k32.SetInformationJobObject(
            job_handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            err = ctypes.get_last_error()
            self._k32.CloseHandle(job_handle)
            raise OSError(err, "SetInformationJobObject failed")
        self._handle = int(job_handle)

    @property
    def active(self) -> bool:
        """Return whether an open Win32 Job Object handle is held."""
        return self._handle is not None

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

    def close(self) -> None:
        """Close the Job Object handle, terminating all assigned processes."""
        if self._k32 is not None and self._handle is not None:
            handle = self._handle
            self._handle = None
            self._k32.CloseHandle(handle)


def pump_stream(stream: IO[bytes], writer: DailyRotatingStreamWriter) -> None:
    """Forward child stdout/stderr lines into a daily rotating stream writer."""
    try:
        for raw_line in iter(stream.readline, b""):
            writer.write_line(raw_line.decode("utf-8", errors="replace"))
    finally:
        stream.close()


def terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Send graceful termination signal to the managed process or process group."""
    if sys.platform != "win32":
        os_killpg = getattr(os, "killpg", None)
        os_getpgid = getattr(os, "getpgid", None)
        if os_killpg is not None and os_getpgid is not None:
            with contextlib.suppress(OSError):
                os_killpg(os_getpgid(proc.pid), signal.SIGTERM)
                return
    proc.terminate()


def kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Forcefully kill the managed process or process group."""
    if sys.platform != "win32":
        os_killpg = getattr(os, "killpg", None)
        os_getpgid = getattr(os, "getpgid", None)
        sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
        if os_killpg is not None and os_getpgid is not None:
            with contextlib.suppress(OSError):
                os_killpg(os_getpgid(proc.pid), sigkill)
                return
    proc.kill()
