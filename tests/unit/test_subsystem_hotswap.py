"""Unit tests for in-place Layer-3 subsystem hot-swapping without state loss."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import respx

from lai.api._cors import DynamicCorsController, DynamicCorsMiddleware
from lai.domain.auth import ApiScope, RequestCredentials
from lai.domain.errors import (
    AuthenticationError,
    AuthorizationError,
    GuardrailBlockedError,
    QueueFullError,
    RateLimitExceededError,
    ValidationFailedError,
)
from lai.domain.guardrails import (
    GuardrailBlockMode,
    GuardrailConfig,
    GuardrailDecision,
    GuardrailVerdict,
)
from lai.domain.scheduler import JobPriority, LeaseRequest
from lai.guardrails import (
    CompositeGuardrailPipeline,
    OllamaShieldGemmaProvider,
    ShieldGemmaCpuProvider,
)
from lai.ingestion import LocalBlobStore
from lai.ports.guardrails import GuardrailProviderPort
from lai.scheduler import FairPriorityQueue
from lai.security import OidcJwtAuthProvider, OidcProviderConfig, TokenBucketRateLimiter
from tests.integration._helpers import OidcTestAuthority

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.types import Message, Receive, Scope, Send


class _AlwaysBlockProvider(GuardrailProviderPort):
    def __init__(self) -> None:
        self.reconfigured: list[GuardrailConfig] = []

    @property
    def name(self) -> str:
        return "always-block"

    def reconfigure(self, config: GuardrailConfig) -> None:
        self.reconfigured.append(config)

    async def evaluate(self, prompt: str) -> GuardrailDecision:
        return GuardrailDecision(
            verdict=GuardrailVerdict.BLOCK,
            provider=self.name,
            violated_policy="test_policy",
            reason="blocked",
            prompt_sha256="a" * 64,
            prompt_chars=len(prompt),
            scrubbed_prompt=prompt,
        )


async def test_oidc_provider_reconfigure_preserves_jwks_cache() -> None:
    """OidcJwtAuthProvider.reconfigure updates audiences/allowlists without re-fetching JWKS."""
    authority = OidcTestAuthority()
    fetch_count = 0

    async def _fetcher(_url: str) -> dict[str, Any]:
        nonlocal fetch_count
        fetch_count += 1
        return {"keys": [authority.jwk]}

    cfg = OidcProviderConfig(
        enabled=True,
        issuer_url=authority.issuer,
        audiences=("client-a",),
        allowed_emails=("alice@example.test",),
    )
    provider = OidcJwtAuthProvider(cfg, jwks_fetcher=_fetcher)

    tok_a = authority.mint_token(sub="u1", email="alice@example.test", aud="client-a")
    p1 = await provider.authenticate(RequestCredentials(bearer_token=tok_a))
    assert p1.principal_id == "oidc:u1"
    assert fetch_count == 1

    tok_b = authority.mint_token(sub="u2", email="bob@example.test", aud="client-b")
    with pytest.raises(AuthenticationError):
        await provider.authenticate(RequestCredentials(bearer_token=tok_b))

    provider.reconfigure(
        cfg.model_copy(
            update={
                "audiences": ("client-a", "client-b"),
                "allowed_emails": ("alice@example.test", "bob@example.test"),
                "admin_emails": ("bob@example.test",),
            }
        )
    )
    p2 = await provider.authenticate(RequestCredentials(bearer_token=tok_b))
    assert p2.scopes == (ApiScope.ADMIN,)
    assert fetch_count == 1

    provider.reconfigure(
        cfg.model_copy(
            update={
                "audiences": ("client-a", "client-b"),
                "allowed_emails": ("bob@example.test",),
            }
        )
    )
    with pytest.raises(AuthorizationError):
        await provider.authenticate(RequestCredentials(bearer_token=tok_a))
    assert fetch_count == 1


async def test_rate_limiter_queue_and_blob_store_reconfigure(tmp_path: Path) -> None:
    """Rate limiter preserves buckets, FairPriorityQueue preserves queued jobs, blob updates max."""
    now = 100.0
    limiter = TokenBucketRateLimiter(requests_per_minute=2, clock=lambda: now)
    await limiter.acquire("u1")
    await limiter.acquire("u1")
    with pytest.raises(RateLimitExceededError):
        await limiter.acquire("u1")

    limiter.reconfigure(requests_per_minute=10)
    with pytest.raises(RateLimitExceededError):
        await limiter.acquire("u1")
    now += 6.0
    await limiter.acquire("u1")

    await limiter.acquire("u2")
    limiter.reconfigure(requests_per_minute=1, burst_capacity=1)
    await limiter.acquire("u2")
    with pytest.raises(RateLimitExceededError):
        await limiter.acquire("u2")

    with pytest.raises(ValueError, match="requests_per_minute"):
        limiter.reconfigure(requests_per_minute=0)
    with pytest.raises(ValueError, match="burst_capacity"):
        limiter.reconfigure(requests_per_minute=5, burst_capacity=0)

    queue = FairPriorityQueue(max_queued_jobs=5, max_queued_jobs_per_principal=3)
    req1 = LeaseRequest(
        session_id="s1", project_id="p1", principal_id="u1", priority=JobPriority.INTERACTIVE
    )
    req2 = LeaseRequest(
        session_id="s2", project_id="p1", principal_id="u1", priority=JobPriority.INTERACTIVE
    )
    req3 = LeaseRequest(
        session_id="s3", project_id="p1", principal_id="u1", priority=JobPriority.INTERACTIVE
    )
    queue.enqueue(req1)
    queue.enqueue(req2)
    queue.reconfigure(max_queued_jobs=2, max_queued_jobs_per_principal=2)
    assert len(queue) == 2
    assert (queue.max_queued_jobs, queue.max_queued_jobs_per_principal) == (2, 2)
    with pytest.raises(QueueFullError):
        queue.enqueue(req3)
    with pytest.raises(ValidationFailedError):
        queue.reconfigure(max_queued_jobs=0, max_queued_jobs_per_principal=1)
    with pytest.raises(ValidationFailedError):
        queue.reconfigure(max_queued_jobs=2, max_queued_jobs_per_principal=0)

    store = LocalBlobStore(tmp_path / "blobs", max_upload_bytes=1024)
    store.set_max_upload_bytes(2048)
    assert store.max_upload_bytes == 2048
    with pytest.raises(ValidationFailedError):
        store.set_max_upload_bytes(0)


async def test_guardrail_providers_and_dynamic_cors_reconfigure(tmp_path: Path) -> None:
    """Guardrail pipeline/providers and DynamicCorsMiddleware hot-swap behavior in place."""
    stub = _AlwaysBlockProvider()
    cfg = GuardrailConfig(enabled=True, block_mode=GuardrailBlockMode.REFUSAL_MESSAGE)
    pipeline = CompositeGuardrailPipeline(cfg, providers=[stub], log_dir=tmp_path)
    try:
        assert (await pipeline.check_prompt("test")).verdict is GuardrailVerdict.BLOCK
        cfg_err = cfg.model_copy(
            update={"block_mode": GuardrailBlockMode.HTTP_ERROR, "refusal_text": "Custom block"}
        )
        pipeline.reconfigure(cfg_err)
        assert stub.reconfigured == [cfg_err]
        with pytest.raises(GuardrailBlockedError, match="Custom block"):
            await pipeline.check_prompt("test")

        pipeline.reconfigure(cfg_err.model_copy(update={"enabled": False}))
        assert (await pipeline.check_prompt("test")).verdict is GuardrailVerdict.ALLOW
    finally:
        pipeline.close()

    sg = ShieldGemmaCpuProvider(GuardrailConfig(port=18082, timeout_seconds=5.0))
    sg.reconfigure(GuardrailConfig(port=18082, timeout_seconds=12.0))
    assert sg.base_url == "http://127.0.0.1:18082"

    og = OllamaShieldGemmaProvider(GuardrailConfig(backend="ollama", violation_threshold=0.90))
    og.reconfigure(GuardrailConfig(backend="ollama", violation_threshold=0.40))
    lp = [{"top_logprobs": [{"token": "Yes", "logprob": -0.5}, {"token": "No", "logprob": -0.9}]}]
    with respx.mock(assert_all_called=True) as mock:
        mock.post("http://127.0.0.1:11434/api/generate").respond(
            200, json={"response": "Yes", "logprobs": lp}
        )
        assert (await og.evaluate("hello")).verdict is GuardrailVerdict.BLOCK

    async def _echo_app(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    ctrl = DynamicCorsController()
    mw = DynamicCorsMiddleware(_echo_app, controller=ctrl)
    seen: list[dict[bytes, bytes]] = []

    async def _call(origin: str) -> None:
        async def _recv() -> Message:
            return {"type": "http.request", "body": b""}

        async def _send(msg: Message) -> None:
            if msg["type"] == "http.response.start":
                seen.append({k.lower(): v for k, v in msg.get("headers", [])})

        scope: Scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/health",
            "headers": [(b"origin", origin.encode("ascii"))],
        }
        await mw(scope, _recv, _send)

    await _call("https://spa.example.test")
    assert b"access-control-allow-origin" not in seen[-1]
    ctrl.reconfigure(("https://spa.example.test",), ("X-Extra",))
    await _call("https://spa.example.test")
    assert seen[-1][b"access-control-allow-origin"] == b"https://spa.example.test"
    ctrl.reconfigure((), ())
    await _call("https://spa.example.test")
    assert b"access-control-allow-origin" not in seen[-1]
