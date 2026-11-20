"""HTTP JSON/verdict/health parsing helpers for ShieldGemmaCpuProvider."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from lai.domain.errors import InferenceBackendError
from lai.domain.inference import BackendHealth
from lai.domain.observability import AlertCategory
from lai.guardrails.prompts import (
    compute_yes_probability,
    has_probability_entries,
    parse_shieldgemma_verdict,
)

if TYPE_CHECKING:
    import httpx
    from structlog.stdlib import BoundLogger

HTTP_OK = 200
_HTTP_BAD_REQUEST = 400
_HEALTHY_STATUSES: frozenset[str] = frozenset({"ok", "ready", "healthy"})
_N_PREDICT = 4
_N_PROBS = 5
_STOP_TOKENS: tuple[str, ...] = ("\n", "<end_of_turn>")


def build_completion_payload(rendered_prompt: str) -> dict[str, object]:
    """Build deterministic ShieldGemma /completion JSON payload with n_probs=5."""
    return {
        "prompt": rendered_prompt,
        "n_predict": _N_PREDICT,
        "n_probs": _N_PROBS,
        "temperature": 0.0,
        "stop": list(_STOP_TOKENS),
    }


def raise_timeout_error(  # noqa: PLR0913
    exc: Exception,
    *,
    provider_name: str,
    principle_name: str,
    timeout_seconds: float,
    prompt_sha256: str,
    log: BoundLogger,
) -> InferenceBackendError:
    """Log TIMEOUT alert and construct InferenceBackendError for ShieldGemma timeouts."""
    log.error(
        "shieldgemma_request_timeout",
        provider=provider_name,
        principle=principle_name,
        timeout_seconds=timeout_seconds,
        prompt_sha256=prompt_sha256,
        alert_category=AlertCategory.TIMEOUT.value,
        exception_type=type(exc).__name__,
    )
    msg = f"ShieldGemma guardrail request timed out after {timeout_seconds}s"
    return InferenceBackendError(
        msg, details={"provider": provider_name, "principle": principle_name}
    )


def raise_http_error(
    exc: Exception,
    *,
    provider_name: str,
    principle_name: str,
    prompt_sha256: str,
    log: BoundLogger,
) -> InferenceBackendError:
    """Log transport/status failure and construct InferenceBackendError."""
    log.error(
        "shieldgemma_http_error",
        provider=provider_name,
        principle=principle_name,
        error=str(exc),
        prompt_sha256=prompt_sha256,
    )
    msg = f"ShieldGemma guardrail request failed: {exc}"
    return InferenceBackendError(
        msg, details={"provider": provider_name, "principle": principle_name}
    )


def parse_json_object(response: httpx.Response) -> dict[str, object] | None:
    """Decode an HTTP response body into a string-keyed dict if valid JSON object."""
    try:
        parsed: object = response.json()
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    raw_map = cast("dict[object, object]", parsed)
    return {str(key): value for key, value in raw_map.items()}


def non_negative_int(value: object) -> int:
    """Coerce integer counts from backend health JSON safely."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def is_context_size_error(response: httpx.Response) -> bool:
    """Return True when llama-server rejected the request for exceeding slot ctx_size."""
    if response.status_code != _HTTP_BAD_REQUEST:
        return False
    payload = parse_json_object(response)
    if payload is not None:
        err = payload.get("error")
        if isinstance(err, dict):
            err_map = cast("dict[object, object]", err)
            err_type = str(err_map.get("type", "")).lower()
            err_msg = str(err_map.get("message", "")).lower()
            if "exceed_context_size" in err_type or "context size" in err_msg:
                return True
        elif isinstance(err, str) and (
            "exceed_context_size" in err.lower() or "context size" in err.lower()
        ):
            return True
    lowered = response.text.lower()
    return "exceed_context_size" in lowered or "exceeds the available context size" in lowered


def parse_completion_response(  # noqa: PLR0913
    response: httpx.Response,
    *,
    provider_name: str,
    principle_name: str,
    violation_threshold: float,
    prompt_sha256: str,
    log: BoundLogger,
) -> tuple[bool, str, float | None]:
    """Validate Yes/No verdict text first, then calibrate using completion_probabilities."""
    data = parse_json_object(response)
    raw_content = data.get("content") if data is not None else None
    if not isinstance(raw_content, str):
        msg = "ShieldGemma response missing string 'content' field."
        raise InferenceBackendError(
            msg,
            details={"provider": provider_name, "principle": principle_name},
        )

    try:
        raw_verdict = parse_shieldgemma_verdict(raw_content)
    except ValueError as exc:
        log.exception(
            "shieldgemma_ambiguous_verdict",
            provider=provider_name,
            principle=principle_name,
            raw_output=raw_content,
            prompt_sha256=prompt_sha256,
        )
        msg = f"Ambiguous ShieldGemma verdict for '{principle_name}': {raw_content!r}"
        raise InferenceBackendError(
            msg,
            details={"provider": provider_name, "principle": principle_name},
        ) from exc

    raw_probs = data.get("completion_probabilities") if data is not None else None
    yes_prob = compute_yes_probability(raw_probs)
    if yes_prob is not None:
        return yes_prob >= violation_threshold, raw_content, yes_prob

    if has_probability_entries(raw_probs):
        if raw_verdict:
            return True, raw_content, None
        log.error(
            "shieldgemma_low_probability_mass",
            provider=provider_name,
            principle=principle_name,
            raw_output=raw_content,
            prompt_sha256=prompt_sha256,
        )
        msg = (
            f"ShieldGemma returned insufficient Yes/No probability mass for '{principle_name}': "
            f"{raw_content!r}"
        )
        raise InferenceBackendError(
            msg,
            details={"provider": provider_name, "principle": principle_name},
        )

    return raw_verdict, raw_content, None


def parse_health_response(
    response: httpx.Response,
    *,
    model_loaded: str,
    elapsed_ms: float,
) -> BackendHealth:
    """Convert a ShieldGemma llama-server /health HTTP response into BackendHealth."""
    payload = parse_json_object(response) or {}
    if response.status_code != HTTP_OK:
        error_obj = payload.get("error")
        error_msg = (
            str(error_obj)
            if error_obj is not None
            else (response.text.strip() or f"HTTP {response.status_code}")
        )
        return BackendHealth(
            healthy=False,
            status=f"http_{response.status_code}",
            model_loaded=model_loaded,
            latency_ms=elapsed_ms,
            error=error_msg,
        )

    status_val = str(payload.get("status", "ok"))
    slots_idle = non_negative_int(payload.get("slots_idle"))
    slots_processing = non_negative_int(payload.get("slots_processing"))
    slots_total = non_negative_int(payload.get("slots_total")) or (slots_idle + slots_processing)
    return BackendHealth(
        healthy=status_val.lower() in _HEALTHY_STATUSES,
        status=status_val,
        model_loaded=model_loaded,
        slots_total=slots_total,
        slots_idle=slots_idle,
        latency_ms=elapsed_ms,
    )
