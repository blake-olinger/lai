"""Local llama.cpp and OpenAI-compatible inference provider and supervisor layer."""

from __future__ import annotations

from lai.inference.downloader import ModelDownloader
from lai.inference.flags import build_llama_server_args
from lai.inference.llamacpp_client import LlamaCppProvider
from lai.inference.ollama_manager import (
    OllamaSupervisor,
    build_ollama_modelfile,
    build_ollama_serve_env,
    derive_hf_quant_tag,
    resolve_ollama_executable,
    setup_ollama_models,
)
from lai.inference.openai_client import OpenAICompatProvider
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
    "OPEN_THOUGHT_TAG",
    "GpuTelemetry",
    "LlamaCppProvider",
    "LlamaCppSupervisor",
    "ModelDownloader",
    "OllamaSupervisor",
    "OpenAICompatProvider",
    "ParsedDelta",
    "ThinkingStreamParser",
    "build_llama_server_args",
    "build_ollama_modelfile",
    "build_ollama_serve_env",
    "derive_hf_quant_tag",
    "query_gpu_telemetry",
    "resolve_ollama_executable",
    "setup_ollama_models",
    "split_thinking_content",
]
