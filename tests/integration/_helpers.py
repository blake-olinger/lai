"""Shared real-service integration builders and ASGI test harnesses."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.profiles import get_model_profile
from lai.domain.auth import ApiScope
from lai.ingestion import DeterministicHashEmbedder, LanceDbVectorStore, LocalIngestionPipeline
from lai.orchestrator import TurnOrchestrator
from lai.security import (
    ApiKeyAuthProvider,
    CompositeAuthenticator,
    OidcJwtAuthProvider,
    OidcProviderConfig,
    build_client_app,
    generate_api_key,
)
from lai.tools import build_default_tool_registry
from tests._helpers import bearer_headers

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from fastapi import FastAPI
    from starlette.types import ASGIApp

    from lai.api.deps import AppServices
    from lai.config.settings import LaiSettings
    from lai.domain.client_apps import ClientAppRecord
    from lai.domain.tools import PermissionTier, ToolSpec
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.orchestrator import OrchestratorPort
    from lai.ports.scheduler import InferenceSpoolerPort
    from lai.ports.tools import ToolPluginPort

_RSA_PUBLIC_EXPONENT = 65537
_RSA_KEY_SIZE = 2048
_DEFAULT_OIDC_ISSUER = "https://accounts.google.com"
_DEFAULT_OIDC_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_DEFAULT_OIDC_SCOPES: tuple[ApiScope, ...] = (
    ApiScope.PROJECTS_READ,
    ApiScope.PROJECTS_WRITE,
    ApiScope.FILES_READ,
    ApiScope.FILES_WRITE,
    ApiScope.CHAT_READ,
    ApiScope.CHAT_WRITE,
    ApiScope.TOOLS_SAFE,
)


@lru_cache(maxsize=1)
def _test_rsa_keypair() -> tuple[rsa.RSAPrivateKey, dict[str, Any]]:
    priv = rsa.generate_private_key(public_exponent=_RSA_PUBLIC_EXPONENT, key_size=_RSA_KEY_SIZE)
    jwk_dict: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(priv.public_key()))
    jwk_dict.update({"kid": "test-oidc-k1", "alg": "RS256", "use": "sig"})
    return priv, jwk_dict


class OidcTestAuthority:
    """In-memory RS256 OIDC JWT issuer wired into AppServices for integration tests."""

    def __init__(self, *, issuer: str = _DEFAULT_OIDC_ISSUER) -> None:
        self._private_key, self.jwk = _test_rsa_keypair()
        self.issuer = issuer

    def mint_token(
        self,
        *,
        sub: str = "oidc-user-1",
        email: str = "dev@acme.test",
        aud: str | list[str] = "test-client",
        azp: str | None = None,
        email_verified: bool = True,
    ) -> str:
        """Sign an RS256 OIDC JWT token with configured claims."""
        now = datetime.now(UTC)
        claims: dict[str, Any] = {
            "iss": self.issuer,
            "aud": aud,
            "sub": sub,
            "email": email,
            "email_verified": email_verified,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=10)).timestamp()),
        }
        if azp is not None:
            claims["azp"] = azp
        return jwt.encode(
            claims,
            self._private_key,
            algorithm="RS256",
            headers={"kid": str(self.jwk["kid"])},
        )

    def mint_headers(
        self,
        *,
        sub: str = "oidc-user-1",
        email: str = "dev@acme.test",
        aud: str | list[str] = "test-client",
        azp: str | None = None,
    ) -> dict[str, str]:
        """Mint an OIDC JWT and return Bearer authorization headers."""
        token = self.mint_token(sub=sub, email=email, aud=aud, azp=azp)
        return bearer_headers(token)


def configure_test_oidc(
    svc: AppServices,
    *,
    audiences: Sequence[str],
    allowed_domains: Sequence[str] = ("acme.test",),
    admin_emails: Sequence[str] = (),
    default_scopes: tuple[ApiScope, ...] = _DEFAULT_OIDC_SCOPES,
) -> OidcTestAuthority:
    """Wire CompositeAuthenticator with ApiKeyAuthProvider and an in-memory OIDC authority."""
    authority = OidcTestAuthority()

    async def _jwks_fetcher(_url: str) -> dict[str, Any]:
        return {"keys": [authority.jwk]}

    oidc_cfg = OidcProviderConfig(
        enabled=True,
        issuer_url=authority.issuer,
        jwks_url=_DEFAULT_OIDC_JWKS_URL,
        audiences=tuple(audiences),
        allowed_domains=tuple(allowed_domains),
        admin_emails=tuple(admin_emails),
        default_scopes=default_scopes,
    )
    svc.authenticator = CompositeAuthenticator(
        providers=[
            ApiKeyAuthProvider(svc.api_key_repo),
            OidcJwtAuthProvider(oidc_cfg, jwks_fetcher=_jwks_fetcher),
        ]
    )
    return authority


@dataclass(slots=True)
class IntegrationSpec:
    """Optional upfront component overrides for an isolated AppServices ASGI harness."""

    orchestrator: OrchestratorPort | None = None
    inference_provider: InferenceProviderPort | None = None
    spooler: InferenceSpoolerPort | None = None
    use_hash_embedder: bool = False
    tools: Sequence[ToolPluginPort] = ()
    raise_app_exceptions: bool = True
    asgi_wrapper: Callable[[ASGIApp], ASGIApp] | None = None


@dataclass(slots=True)
class IntegrationHarness:
    """Running FastAPI + AppServices + httpx.AsyncClient test environment."""

    settings: LaiSettings
    svc: AppServices
    app: FastAPI
    client: httpx.AsyncClient


def _wire_upfront_services(
    svc: AppServices,
    settings: LaiSettings,
    spec: IntegrationSpec,
) -> AppServices:
    """Assemble customized embedder, vector store, tools, inference, and orchestrator upfront."""
    updates: dict[str, Any] = {}
    embedder = svc.embedder
    vector_store = svc.vector_store
    tool_registry = svc.tool_registry

    if spec.use_hash_embedder:
        embedder = DeterministicHashEmbedder()
        vector_store = LanceDbVectorStore(
            settings.data_dir / "lancedb", dimensions=embedder.dimensions
        )
        updates["embedder"] = embedder
        updates["vector_store"] = vector_store
        updates["ingestion_pipeline"] = LocalIngestionPipeline(
            blob_store=svc.blob_store,
            file_repo=svc.file_repo,
            embedder=embedder,
            vector_store=vector_store,
        )
        tool_registry = build_default_tool_registry(
            svc.file_repo, svc.blob_store, embedder=embedder, vector_store=vector_store
        )
        updates["tool_registry"] = tool_registry

    for tool in spec.tools:
        tool_registry.register(tool)

    inference = spec.inference_provider or svc.inference_provider
    spooler = spec.spooler or svc.spooler
    if spec.inference_provider is not None:
        updates["inference_provider"] = spec.inference_provider
    if spec.spooler is not None:
        updates["spooler"] = spec.spooler

    needs_orchestrator = spec.orchestrator is None and (
        spec.use_hash_embedder
        or spec.inference_provider is not None
        or spec.spooler is not None
        or bool(spec.tools)
    )
    if needs_orchestrator:
        profile = get_model_profile(settings.default_model_profile, settings.custom_profiles)
        orch_embedder = embedder if spec.use_hash_embedder else None
        orch_vector_store = vector_store if spec.use_hash_embedder else None
        updates["orchestrator"] = TurnOrchestrator(
            settings=settings,
            session_repo=svc.session_repo,
            file_repo=svc.file_repo,
            blob_store=svc.blob_store,
            inference_provider=inference,
            spooler=spooler,
            session_gate=svc.session_gate,
            guardrail_pipeline=svc.guardrail_pipeline,
            tool_registry=tool_registry,
            tool_executor=svc.tool_executor,
            embedder=orch_embedder,
            vector_store=orch_vector_store,
            model_profile=profile,
            client_app_repo=svc.client_app_repo,
        )

    return replace(svc, **updates) if updates else svc


@asynccontextmanager
async def open_services_harness(
    settings: LaiSettings,
    spec: IntegrationSpec | None = None,
) -> AsyncGenerator[IntegrationHarness]:
    """Build AppServices upfront, mount ASGI client, and close cleanly on exit."""
    resolved = spec or IntegrationSpec()
    base_svc = await build_app_services(settings, orchestrator=resolved.orchestrator)
    svc = _wire_upfront_services(base_svc, settings, resolved)
    app = create_app(settings, services=svc)
    asgi_target: ASGIApp = resolved.asgi_wrapper(app) if resolved.asgi_wrapper is not None else app
    transport = httpx.ASGITransport(
        app=cast("Any", asgi_target),
        raise_app_exceptions=resolved.raise_app_exceptions,
    )
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
        ):
            yield IntegrationHarness(settings=settings, svc=svc, app=app, client=client)
    finally:
        await close_app_services(svc)


async def mint_api_key(
    svc: AppServices,
    name: str,
    scopes: tuple[ApiScope, ...],
) -> tuple[str, str]:
    """Persist a generated API key and return (plaintext_key, key_id)."""
    created = generate_api_key(name, scopes)
    await svc.api_key_repo.create(created.record)
    return created.plaintext_key, created.record.id


async def mint_auth_headers(
    svc: AppServices,
    name: str,
    scopes: tuple[ApiScope, ...],
) -> dict[str, str]:
    """Persist a generated API key and return Bearer authorization headers."""
    plaintext, _ = await mint_api_key(svc, name, scopes)
    return bearer_headers(plaintext)


async def create_project(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    name: str,
) -> str:
    """Create a project via HTTP and return its ID."""
    response = await client.post("/v1/projects", headers=headers, json={"name": name})
    return str(response.json()["id"])


async def create_session(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    project_id: str,
    *,
    title: str | None = None,
) -> str:
    """Create a chat session via HTTP and return its ID."""
    payload: dict[str, object] = {"project_id": project_id}
    if title is not None:
        payload["title"] = title
    response = await client.post("/v1/sessions", headers=headers, json=payload)
    return str(response.json()["id"])


async def register_client_app(  # noqa: PLR0913
    svc: AppServices,
    name: str,
    *,
    oauth_client_id: str,
    allowed_tiers: tuple[PermissionTier, ...],
    registered_tools: tuple[ToolSpec, ...],
    allowed_project_ids: tuple[str, ...] = (),
    expires_at: datetime | None = None,
) -> ClientAppRecord:
    """Persist an OIDC-bound ClientApp record and return it."""
    record = build_client_app(
        name,
        oauth_client_id=oauth_client_id,
        allowed_tiers=allowed_tiers,
        registered_tools=registered_tools,
        allowed_project_ids=allowed_project_ids,
        expires_at=expires_at,
    )
    return await svc.client_app_repo.create(record)
