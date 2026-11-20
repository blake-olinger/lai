"""Pinned Linux x64 llama.cpp CUDA release manifest and model provenance constants."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from lai.domain.inference import LlamaCppBinarySpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# renovate: datasource=github-releases depName=ggml-org/llama.cpp
DEFAULT_LLAMA_CPP_TAG = "b12552"
DEFAULT_CUDA_VERSION = "12.8"
SUPPORTED_CUDA_VERSIONS: tuple[str, ...] = ("12.8", "13.5")

LLAMA_CPP_LINUX_X64_MANIFEST: dict[str, LlamaCppBinarySpec] = {
    "12.8": LlamaCppBinarySpec(
        tag=DEFAULT_LLAMA_CPP_TAG,
        cuda_version="12.8",
        asset_name="llama-b12552-bin-ubuntu-cuda-12.8-x64.tar.gz",
        sha256="911f3b7c79c395b504713ec34a0c12d1a5e6a65654df9146043dc8b6972243f7",
        cudart_asset_name="cudart-llama-b12552-bin-ubuntu-cuda-12.8-x64.tar.gz",
        cudart_sha256="56a1e90b23faeba322b2167fa58aeabce2ddbfa6cdc1683b41dedcac8a96122a",
    ),
    "13.5": LlamaCppBinarySpec(
        tag=DEFAULT_LLAMA_CPP_TAG,
        cuda_version="13.5",
        asset_name="llama-b12552-bin-ubuntu-cuda-13.5-x64.tar.gz",
        sha256="89762cb4d75cefeb09377dc87380c6efa749b33a99dc571883a8f8c73c322859",
        cudart_asset_name="cudart-llama-b12552-bin-ubuntu-cuda-13.5-x64.tar.gz",
        cudart_sha256="b9009633aed4c5529fea6506fb081ed68696c5ff33ddb39d76e985644667c612",
    ),
}

# Pinned live HuggingFace Hub provenance for ShieldGemma-2B guardrail weights.
SHIELDGEMMA_2B_HF_REPO = "QuantFactory/shieldgemma-2b-GGUF"
SHIELDGEMMA_2B_HF_FILENAME = "shieldgemma-2b.Q4_K_M.gguf"
SHIELDGEMMA_2B_HF_REVISION = "8755ece4b20dd84d7564131e385d9c205d03621a"
SHIELDGEMMA_2B_SHA256 = "47b0c3f4ec0bf93659ab2fc92cf2041374ef78bf1cb5b8c790421f463e7b7979"

_CUDA_NUMS_RE = re.compile(r"\d+")

__all__ = [
    "DEFAULT_CUDA_VERSION",
    "DEFAULT_LLAMA_CPP_TAG",
    "LLAMA_CPP_LINUX_X64_MANIFEST",
    "SHIELDGEMMA_2B_HF_FILENAME",
    "SHIELDGEMMA_2B_HF_REPO",
    "SHIELDGEMMA_2B_HF_REVISION",
    "SHIELDGEMMA_2B_SHA256",
    "SUPPORTED_CUDA_VERSIONS",
    "lookup_manifest_sha256",
    "parse_cuda_version_tuple",
    "select_compatible_cuda_version",
]


def parse_cuda_version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse numeric CUDA version components (e.g. '13.5' -> (13, 5))."""
    return tuple(int(part) for part in _CUDA_NUMS_RE.findall(version_str))


def select_compatible_cuda_version(
    driver_cuda: tuple[int, int] | None,
    *,
    supported: Sequence[str] = SUPPORTED_CUDA_VERSIONS,
    default: str = DEFAULT_CUDA_VERSION,
) -> str:
    """Select the newest supported CUDA toolkit <= driver_cuda, defaulting to 12.8."""
    if driver_cuda is None or not supported:
        return default
    compatible = [
        version
        for version in supported
        if (parsed := parse_cuda_version_tuple(version)) and parsed <= driver_cuda
    ]
    if not compatible:
        return default
    return max(compatible, key=parse_cuda_version_tuple)


def lookup_manifest_sha256(
    asset_name: str,
    *,
    manifest: Mapping[str, LlamaCppBinarySpec] = LLAMA_CPP_LINUX_X64_MANIFEST,
) -> str | None:
    """Look up the pinned SHA-256 digest for a llama.cpp or cudart archive filename."""
    for spec in manifest.values():
        if spec.asset_name == asset_name:
            return spec.sha256
        if spec.cudart_asset_name == asset_name:
            return spec.cudart_sha256
    return None
