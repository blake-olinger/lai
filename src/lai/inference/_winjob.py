"""Inference GPU telemetry parsing, log stream pumping, and process-group helpers."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

from pydantic import Field, ValidationError

from lai.domain.base import StrictModel
from lai.observability._process import (
    WindowsJobObject,
)
from lai.observability._process import (
    kill_popen_process_group as kill_process_group,
)
from lai.observability._process import (
    terminate_popen_process_group as terminate_process_group,
)

if TYPE_CHECKING:
    from lai.observability.rotation import DailyRotatingStreamWriter

_MIN_GPU_CSV_FIELDS = 4
_GPU_TEMP_FIELD_INDEX = 4
_MAX_UTIL_PERCENT = 100

__all__ = [
    "GpuTelemetry",
    "WindowsJobObject",
    "kill_process_group",
    "parse_nvidia_smi_output",
    "pump_stream",
    "terminate_process_group",
]


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


def pump_stream(stream: IO[bytes], writer: DailyRotatingStreamWriter) -> None:
    """Forward child stdout/stderr lines into a daily rotating stream writer."""
    try:
        for raw_line in iter(stream.readline, b""):
            writer.write_line(raw_line.decode("utf-8", errors="replace"))
    finally:
        stream.close()
