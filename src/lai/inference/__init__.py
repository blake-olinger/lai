"""Local Linux llama.cpp inference provider, manifest, downloader, and supervisor layer."""

from __future__ import annotations

from lai.inference._runtime_helpers import detect_nvidia_cuda_version
from lai.inference.downloader import ModelDownloader
from lai.inference.flags import build_llama_server_args
from lai.inference.llamacpp_client import LlamaCppProvider
from lai.inference.manifest import (
    DEFAULT_CUDA_VERSION,
    DEFAULT_LLAMA_CPP_TAG,
    LLAMA_CPP_LINUX_X64_MANIFEST,
    select_compatible_cuda_version,
)
from lai.inference.supervisor import GpuTelemetry, LlamaCppSupervisor, query_gpu_telemetry
from lai.inference.thinking import (
    CLOSE_THOUGHT_TAG,
    OPEN_THOUGHT_TAG,
    ParsedDelta,
    ThinkingStreamParser,
    split_thinking_content,
)

__all__ = [
    "CLOSE_THOUGHT_TAG",
    "DEFAULT_CUDA_VERSION",
    "DEFAULT_LLAMA_CPP_TAG",
    "LLAMA_CPP_LINUX_X64_MANIFEST",
    "OPEN_THOUGHT_TAG",
    "GpuTelemetry",
    "LlamaCppProvider",
    "LlamaCppSupervisor",
    "ModelDownloader",
    "ParsedDelta",
    "ThinkingStreamParser",
    "build_llama_server_args",
    "detect_nvidia_cuda_version",
    "query_gpu_telemetry",
    "select_compatible_cuda_version",
    "split_thinking_content",
]
