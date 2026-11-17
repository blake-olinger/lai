"""Built-in model profiles tuned for local llama.cpp hardware constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.errors import NotFoundError
from lai.domain.inference import ModelProfile

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_MODEL_PROFILE_NAME = "gemma4-26b-a4b"

GEMMA4_26B_A4B_PROFILE = ModelProfile(
    name=DEFAULT_MODEL_PROFILE_NAME,
    hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
    fallback_hf_repo="bartowski/google_gemma-4-26B-A4B-it-GGUF",
    hf_filename="gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
    ctx_size=24576,
    flash_attn=True,
    kv_cache_type_k="q8_0",
    kv_cache_type_v="q8_0",
    parallel_slots=1,
    n_gpu_layers=999,
    cpu_moe_layers=3,
    batch_size=2048,
    ubatch_size=512,
    jinja=True,
    reasoning_format="deepseek",
    temperature=0.2,
    top_p=0.95,
    max_tokens=4096,
)

BUILTIN_PROFILES: dict[str, ModelProfile] = {
    DEFAULT_MODEL_PROFILE_NAME: GEMMA4_26B_A4B_PROFILE,
}


def get_model_profile(
    name: str,
    custom_profiles: Mapping[str, ModelProfile] | None = None,
) -> ModelProfile:
    """Resolve a model profile by name, preferring custom overrides."""
    if custom_profiles is not None and name in custom_profiles:
        return custom_profiles[name]
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]
    msg = f"Unknown model profile: {name!r}"
    raise NotFoundError(msg, details={"profile": name})
