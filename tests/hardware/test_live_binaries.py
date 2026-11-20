"""Opt-in hardware tests exercising live ONNX embeddings, llama-server, and ShieldGemma-2B."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import pytest

from lai.cli._common import resolve_llama_executable
from lai.config.profiles import get_model_profile
from lai.config.settings import load_settings
from lai.domain.guardrails import GuardrailVerdict
from lai.guardrails import ShieldGemmaCpuProvider, ShieldGemmaSupervisor
from lai.inference import LlamaCppProvider, LlamaCppSupervisor
from lai.ingestion.embedder import DEFAULT_EMBEDDING_DIMENSIONS, FastEmbedCpuAdapter

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.hardware

_HARDWARE_ALLOW_DOWNLOAD_ENV = "LAI_HARDWARE_ALLOW_DOWNLOAD"
_L2_NORM_TOL = 1e-3
_SIMILARITY_DOC_COUNT = 2


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


@pytest.mark.hardware
async def test_live_shieldgemma_supervisor_and_provider(tmp_path: Path) -> None:
    """Verify live ShieldGemmaSupervisor + ShieldGemmaCpuProvider when binary and GGUF exist."""
    settings = load_settings()
    exe_path = resolve_llama_executable(settings.bin_dir)
    if not exe_path.exists():
        pytest.skip(f"llama-server executable not found at {exe_path}")

    guardrail_model_path = settings.models_dir / settings.guardrails.hf_filename
    if not guardrail_model_path.exists():
        pytest.skip(f"ShieldGemma GGUF model not found at {guardrail_model_path}")

    supervisor = ShieldGemmaSupervisor(
        executable_path=exe_path,
        model_path=guardrail_model_path,
        log_dir=tmp_path / "shieldgemma-logs",
    )
    provider = ShieldGemmaCpuProvider(settings.guardrails)
    try:
        sup_health = await supervisor.ensure_running(settings.guardrails)
        prov_health = await provider.health()
        assert sup_health.healthy
        assert prov_health.healthy
        decision = await provider.evaluate("Summarize the key features of Python 3.13.")
        assert decision.verdict is GuardrailVerdict.ALLOW
    finally:
        await supervisor.stop()
