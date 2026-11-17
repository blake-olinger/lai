"""Project CRUD routes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from lai.api.deps import AppServices, get_services, require_scope_dep
from lai.api.schemas.common import DeleteResponse
from lai.api.schemas.projects import ProjectCreateRequest, ProjectListResponse
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.errors import NotFoundError
from lai.domain.projects import Project

router = APIRouter(prefix="/v1/projects", tags=["projects"])

_MAX_CLEANUP_FILES = 10_000


@router.post("", response_model=Project, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreateRequest,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.PROJECTS_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
) -> Project:
    """Create a new isolated project workspace."""
    project = Project(
        name=body.name,
        description=body.description,
        workspace_root=body.workspace_root,
        default_model_profile=body.default_model_profile,
    )
    return await services.project_repo.create(project)


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.PROJECTS_READ))],
    services: Annotated[AppServices, Depends(get_services)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ProjectListResponse:
    """List workspace projects."""
    items = await services.project_repo.list_all(limit=limit, offset=offset)
    return ProjectListResponse(items=items, count=len(items))


@router.get("/{project_id}", response_model=Project)
async def get_project(
    project_id: str,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.PROJECTS_READ))],
    services: Annotated[AppServices, Depends(get_services)],
) -> Project:
    """Fetch a project by ID."""
    project = await services.project_repo.get(project_id)
    if project is None:
        msg = f"Project {project_id!r} not found."
        raise NotFoundError(msg, details={"project_id": project_id})
    return project


@router.delete("/{project_id}", response_model=DeleteResponse)
async def delete_project(
    project_id: str,
    _principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.PROJECTS_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
) -> DeleteResponse:
    """Delete a project along with its stored file blobs, vectors, and metadata."""
    project = await services.project_repo.get(project_id)
    if project is None:
        msg = f"Project {project_id!r} not found."
        raise NotFoundError(msg, details={"project_id": project_id})

    files = await services.file_repo.list_files(project_id, limit=_MAX_CLEANUP_FILES)
    for record in files:
        if services.vector_store is not None:
            await services.vector_store.delete_file_vectors(project_id, record.id)
        await services.blob_store.delete_blob(Path(record.blob_path))

    deleted = await services.project_repo.delete(project_id)
    return DeleteResponse(id=project_id, deleted=deleted)
