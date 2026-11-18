"""Shared real-service integration builders and ASGI test harnesses."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import httpx

from lai.api.app import create_app
from lai.api.runtime import build_app_services, close_app_services
from lai.config.profiles import get_model_profile
from lai.ingestion import DeterministicHashEmbedder, LanceDbVectorStore, LocalIngestionPipeline
from lai.orchestrator import TurnOrchestrator
from lai.security import generate_api_key, generate_client_app
from lai.tools import build_default_tool_registry
from tests._helpers import bearer_headers

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from fastapi import FastAPI
    from starlette.types import ASGIApp

    from lai.api.deps import AppServices
    from lai.config.settings import LaiSettings
    from lai.domain.auth import ApiScope
    from lai.domain.tools import PermissionTier, ToolSpec
    from lai.ports.inference import InferenceProviderPort
    from lai.ports.orchestrator import OrchestratorPort
    from lai.ports.scheduler import InferenceSpoolerPort
    from lai.ports.tools import ToolPluginPort


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


async def register_client_app(
    svc: AppServices,
    name: str,
    *,
    allowed_tiers: tuple[PermissionTier, ...],
    registered_tools: tuple[ToolSpec, ...],
) -> tuple[str, str]:
    """Persist a ClientApp record and return (app_id, plaintext_secret)."""
    created = generate_client_app(
        name,
        allowed_tiers=allowed_tiers,
        registered_tools=registered_tools,
    )
    await svc.client_app_repo.create(created.record)
    return created.record.id, created.plaintext_secret
