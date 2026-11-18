"""Opt-in hardware tests exercising live ONNX embeddings and local inference binaries."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import pytest

from lai.cli._common import resolve_llama_executable
from lai.config.profiles import get_model_profile
from lai.config.settings import load_settings
from lai.domain.chat import MessageRole
from lai.domain.errors import InferenceBackendError
from lai.domain.inference import BackendHealth, InferenceMessage, InferenceRequest
from lai.inference import (
    LlamaCppProvider,
    LlamaCppSupervisor,
    OllamaSupervisor,
    OpenAICompatProvider,
)
from lai.inference.ollama_manager import resolve_ollama_executable
from lai.ingestion.embedder import DEFAULT_EMBEDDING_DIMENSIONS, FastEmbedCpuAdapter

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.hardware

_HARDWARE_ALLOW_DOWNLOAD_ENV = "LAI_HARDWARE_ALLOW_DOWNLOAD"
_L2_NORM_TOL = 1e-3
_SIMILARITY_DOC_COUNT = 2
_SAMPLE_CHAT_MAX_TOKENS = 8


def _dot(vec_a: list[float], vec_b: list[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b, strict=True))


@pytest.mark.hardware
async def test_live_fastembed_cpu_embeddings() -> None:
    """Verify real FastEmbedCpuAdapter ONNX CPU document and query embeddings when cached."""
    settings = load_settings()
    fastembed_dir = settings.models_dir / "fastembed"
    cache_dir = fastembed_dir if fastembed_dir.exists() else None
    local_only = os.environ.get(_HARDWARE_ALLOW_DOWNLOAD_ENV) != "1"

    adapter = FastEmbedCpuAdapter(
        cache_dir=cache_dir,
        local_files_only=local_only,
    )
    docs = [
        "PostgreSQL connection pool timeout exception on port 5432",
        "CSS flexbox layout padding and responsive grid margins",
    ]
    try:
        doc_vectors = await adapter.embed_texts(docs)
        query_vector = await adapter.embed_query("database connection timeout")
    except (ValueError, OSError, RuntimeError) as exc:
        pytest.skip(f"FastEmbed ONNX cache not available offline: {exc}")

    assert len(doc_vectors) == _SIMILARITY_DOC_COUNT
    for vec in (*doc_vectors, query_vector):
        assert len(vec) == DEFAULT_EMBEDDING_DIMENSIONS
        norm = math.sqrt(sum(val * val for val in vec))
        assert math.isclose(norm, 1.0, rel_tol=_L2_NORM_TOL)

    assert _dot(doc_vectors[0], query_vector) > _dot(doc_vectors[1], query_vector)


@pytest.mark.hardware
async def test_live_ollama_supervisor_and_provider_health(tmp_path: Path) -> None:
    """Probe live OllamaSupervisor and OpenAICompatProvider health when ollama is installed."""
    try:
        ollama_exe = resolve_ollama_executable()
    except InferenceBackendError as exc:
        pytest.skip(f"Ollama binary not installed on host: {exc.message}")

    settings = load_settings()
    supervisor = OllamaSupervisor(
        base_url=settings.ollama_base_url,
        log_dir=tmp_path / "ollama-logs",
        executable_path=ollama_exe,
    )
    provider = OpenAICompatProvider(settings.ollama_base_url, health_timeout_seconds=2.0)
    try:
        sup_health = await supervisor.health()
        prov_health = await provider.health()
    finally:
        await provider.aclose()

    assert isinstance(sup_health, BackendHealth)
    assert isinstance(prov_health, BackendHealth)
    assert sup_health.status
    assert prov_health.status
    assert sup_health.latency_ms is not None
    assert sup_health.latency_ms >= 0.0
    assert prov_health.latency_ms is not None
    assert prov_health.latency_ms >= 0.0
    if sup_health.healthy:
        assert prov_health.healthy


@pytest.mark.hardware
async def test_live_ollama_chat_completion_when_daemon_ready(tmp_path: Path) -> None:
    """Run a tiny live chat completion when an Ollama daemon and model are already running."""
    try:
        ollama_exe = resolve_ollama_executable()
    except InferenceBackendError as exc:
        pytest.skip(f"Ollama binary not installed on host: {exc.message}")

    settings = load_settings()
    supervisor = OllamaSupervisor(
        base_url=settings.ollama_base_url,
        log_dir=tmp_path / "ollama-chat-logs",
        executable_path=ollama_exe,
    )
    health = await supervisor.health()
    if not health.healthy or not health.model_loaded:
        pytest.skip(
            f"Live Ollama daemon not ready with a loaded model ({health.status}): {health.error}"
        )

    provider = OpenAICompatProvider(settings.ollama_base_url, timeout_seconds=30.0)
    try:
        request = InferenceRequest(
            model=health.model_loaded,
            messages=(InferenceMessage(role=MessageRole.USER, content="Reply with OK."),),
            max_tokens=_SAMPLE_CHAT_MAX_TOKENS,
            temperature=0.0,
        )
        response = await provider.chat_complete(request)
    finally:
        await provider.aclose()

    assert response.content.strip() or response.thinking.strip()


@pytest.mark.hardware
async def test_live_llamacpp_supervisor_and_provider(tmp_path: Path) -> None:
    """Verify live LlamaCppSupervisor + LlamaCppProvider when binary and GGUF model exist."""
    settings = load_settings()
    exe_path = resolve_llama_executable(settings.bin_dir)
    if not exe_path.exists():
        pytest.skip(f"llama-server executable not found at {exe_path}")

    profile = get_model_profile(settings.default_model_profile, settings.custom_profiles)
    model_path = settings.models_dir / profile.hf_filename
    if not model_path.exists():
        pytest.skip(f"GGUF model file not found at {model_path}")

    supervisor = LlamaCppSupervisor(
        executable_path=exe_path,
        models_dir=settings.models_dir,
        log_dir=tmp_path / "llama-logs",
        host=settings.llama_host,
        port=settings.llama_port,
    )
    provider = LlamaCppProvider(
        host=settings.llama_host,
        port=settings.llama_port,
        health_timeout_seconds=2.0,
    )
    try:
        sup_health = await supervisor.ensure_running(profile)
        prov_health = await provider.health()
        assert sup_health.healthy
        assert prov_health.healthy
        assert supervisor.active_profile() == profile
    finally:
        await provider.aclose()
        await supervisor.stop()
