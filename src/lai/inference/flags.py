"""Command-line argument builder for the supervised llama-server process."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.errors import ValidationFailedError

if TYPE_CHECKING:
    from pathlib import Path

    from lai.domain.inference import ModelProfile

_MIN_PORT = 1
_MAX_PORT = 65535


def build_llama_server_args(
    profile: ModelProfile,
    model_path: Path,
    host: str,
    port: int,
) -> list[str]:
    """Build llama-server CLI flags from a hardware-tuned ModelProfile."""
    if not host.strip():
        msg = "llama-server host must not be empty."
        raise ValidationFailedError(msg, details={"host": host})
    if port < _MIN_PORT or port > _MAX_PORT:
        msg = f"llama-server port must be between {_MIN_PORT} and {_MAX_PORT}: {port}"
        raise ValidationFailedError(msg, details={"port": str(port)})

    args: list[str] = [
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--parallel",
        str(profile.parallel_slots),
        "--ctx-size",
        str(profile.ctx_size),
    ]
    if profile.flash_attn:
        args.extend(["--flash-attn", "on"])
    if profile.jinja:
        args.append("--jinja")

    args.extend(
        [
            "-ctk",
            profile.kv_cache_type_k,
            "-ctv",
            profile.kv_cache_type_v,
            "-ngl",
            str(profile.n_gpu_layers),
        ]
    )
    if profile.cpu_moe_layers > 0:
        args.extend(["--n-cpu-moe", str(profile.cpu_moe_layers)])

    args.extend(
        [
            "-b",
            str(profile.batch_size),
            "-ub",
            str(profile.ubatch_size),
            "--no-mmap",
        ]
    )
    if profile.reasoning_format:
        args.extend(["--reasoning-format", profile.reasoning_format])
    return args
