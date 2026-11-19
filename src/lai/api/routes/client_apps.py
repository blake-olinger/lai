"""Admin REST routes for managing OIDC-bound registered client applications."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from lai.api.deps import AppServices, get_services, require_scope_dep
from lai.api.schemas.client_apps import (
    ClientAppListResponse,
    ClientAppResponse,
    CreateClientAppRequest,
    UpdateClientAppRequest,
)
from lai.api.schemas.common import DeleteResponse
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.base import utc_now
from lai.domain.errors import LaiError, NotFoundError
from lai.ports.repositories import ClientAppRepositoryPort
from lai.security import build_client_app, update_client_app

router = APIRouter(prefix="/v1/client-apps", tags=["client-apps"])


def _get_client_app_repo(services: AppServices) -> ClientAppRepositoryPort:
    """Resolve ClientAppRepositoryPort from AppServices cleanly."""
    repo = getattr(services, "client_app_repo", None)
    if not isinstance(repo, ClientAppRepositoryPort):
        msg = "Client app repository is not configured."
        raise LaiError(msg, code="service_unavailable", status_code=503)
    return repo


@router.post("", response_model=ClientAppResponse, status_code=status.HTTP_201_CREATED)
async def create_client_app(
    body: CreateClientAppRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> ClientAppResponse:
    """Register a new client application bound to an OAuth client ID."""
    repo = _get_client_app_repo(services)
    record = build_client_app(
        name=body.name,
        oauth_client_id=body.oauth_client_id,
        allowed_tiers=body.allowed_tiers,
        registered_tools=body.registered_tools,
        allowed_project_ids=body.allowed_project_ids,
        expires_at=body.expires_at,
    )
    saved = await repo.create(record)
    return ClientAppResponse.from_record(saved)


@router.get("", response_model=ClientAppListResponse)
async def list_client_apps(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> ClientAppListResponse:
    """List all registered client applications and their bound OAuth client IDs."""
    repo = _get_client_app_repo(services)
    records = await repo.list_all()
    now = utc_now()
    items = [ClientAppResponse.from_record(record, now=now) for record in records]
    return ClientAppListResponse(items=items, count=len(items))


@router.patch("/{app_id}", response_model=ClientAppResponse)
async def patch_client_app(
    app_id: str,
    body: UpdateClientAppRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> ClientAppResponse:
    """Update mutable settings on an active (non-revoked) client application."""
    repo = _get_client_app_repo(services)
    existing = await repo.get_by_id(app_id)
    if existing is None:
        msg = f"Client app {app_id!r} not found."
        raise NotFoundError(msg, details={"app_id": app_id})
    updated_record = update_client_app(
        existing,
        name=body.name,
        allowed_tiers=body.allowed_tiers,
        registered_tools=body.registered_tools,
        allowed_project_ids=body.allowed_project_ids,
        expires_at=body.expires_at,
        update_expires_at="expires_at" in body.model_fields_set,
    )
    saved = await repo.update(updated_record)
    return ClientAppResponse.from_record(saved)


@router.delete("/{app_id}", response_model=DeleteResponse)
async def revoke_client_app(
    app_id: str,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.ADMIN))],
    services: Annotated[AppServices, Depends(get_services)],
) -> DeleteResponse:
    """Revoke an existing client application by ID."""
    repo = _get_client_app_repo(services)
    revoked = await repo.revoke(app_id, utc_now())
    if not revoked:
        msg = f"Client app {app_id!r} not found."
        raise NotFoundError(msg, details={"app_id": app_id})
    return DeleteResponse(id=app_id, deleted=True)
