"""Multi-tenant ownership checks and per-principal project/storage quota enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lai.domain.auth import ApiScope
from lai.domain.errors import AuthorizationError, NotFoundError, PayloadTooLargeError
from lai.security import has_effective_scope

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from lai.api.deps import AppServices
    from lai.domain.auth import AuthPrincipal
    from lai.domain.chat import Session
    from lai.domain.files import FileRecord
    from lai.domain.projects import Project
    from lai.domain.tools import ToolApprovalRecord


def is_admin_principal(principal: AuthPrincipal) -> bool:
    """Return True when the principal holds effective ADMIN scope."""
    return has_effective_scope(principal, ApiScope.ADMIN)


async def require_accessible_project(
    services: AppServices,
    project_id: str,
    principal: AuthPrincipal,
) -> Project:
    """Fetch project_id or raise 404 NotFoundError if missing or owned by another principal."""
    project = await services.project_repo.get(project_id)
    if project is None or (
        not is_admin_principal(principal) and project.owner_principal_id != principal.principal_id
    ):
        msg = f"Project {project_id!r} not found."
        raise NotFoundError(msg, details={"project_id": project_id})
    return project


async def require_accessible_session(
    services: AppServices,
    session_id: str,
    principal: AuthPrincipal,
) -> Session:
    """Fetch session_id or raise 404 NotFoundError if missing or its project is inaccessible."""
    session = await services.session_repo.get_session(session_id)
    if session is None:
        msg = f"Session {session_id!r} not found."
        raise NotFoundError(msg, details={"session_id": session_id})
    try:
        await require_accessible_project(services, session.project_id, principal)
    except NotFoundError as exc:
        msg = f"Session {session_id!r} not found."
        raise NotFoundError(msg, details={"session_id": session_id}) from exc
    return session


async def require_accessible_file(
    services: AppServices,
    file_id: str,
    principal: AuthPrincipal,
) -> FileRecord:
    """Fetch file_id or raise 404 NotFoundError if missing or its project is inaccessible."""
    record = await services.file_repo.get_file(file_id)
    if record is None:
        msg = f"File {file_id!r} not found."
        raise NotFoundError(msg, details={"file_id": file_id})
    try:
        await require_accessible_project(services, record.project_id, principal)
    except NotFoundError as exc:
        msg = f"File {file_id!r} not found."
        raise NotFoundError(msg, details={"file_id": file_id}) from exc
    return record


async def require_project_files(
    services: AppServices,
    project_id: str,
    file_ids: Sequence[str],
    principal: AuthPrincipal,
) -> None:
    """Ensure every file_id is accessible to principal (404) and belongs to project_id (403)."""
    for file_id in dict.fromkeys(file_ids):
        record = await require_accessible_file(services, file_id, principal)
        if record.project_id != project_id:
            msg = f"File {file_id!r} does not belong to project {project_id!r}."
            raise AuthorizationError(
                msg,
                details={"file_id": file_id, "project_id": project_id},
            )


async def require_accessible_approval(
    services: AppServices,
    approval_id: str,
    principal: AuthPrincipal,
) -> ToolApprovalRecord:
    """Fetch approval_id or raise 404 NotFoundError if missing or its project is inaccessible."""
    record = await services.tool_audit_repo.get_approval(approval_id)
    if record is None:
        msg = f"Tool approval {approval_id!r} not found."
        raise NotFoundError(msg, details={"approval_id": approval_id})
    try:
        await require_accessible_project(services, record.project_id, principal)
    except NotFoundError as exc:
        msg = f"Tool approval {approval_id!r} not found."
        raise NotFoundError(msg, details={"approval_id": approval_id}) from exc
    return record


async def enforce_project_quota(services: AppServices, principal: AuthPrincipal) -> None:
    """Enforce per-principal project limit (admins exempt), raising 403 on exhaustion."""
    if is_admin_principal(principal):
        return
    count = await services.project_repo.count_for_owner(principal.principal_id)
    limit = services.settings.max_projects_per_principal
    if count >= limit:
        msg = f"Principal {principal.principal_id!r} reached the maximum project quota ({limit})."
        raise AuthorizationError(
            msg,
            code="quota_exceeded",
            details={
                "max_projects_per_principal": str(limit),
                "current_projects": str(count),
            },
        )


async def enforce_and_wrap_storage_quota(
    services: AppServices,
    principal: AuthPrincipal,
    stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Check pre-upload byte usage and wrap stream to abort if quota is exceeded mid-upload."""
    if is_admin_principal(principal):
        return stream
    used = await services.file_repo.sum_bytes_for_owner(principal.principal_id)
    limit = services.settings.max_storage_bytes_per_principal
    if used >= limit:
        msg = f"Principal {principal.principal_id!r} exceeded storage quota of {limit} bytes."
        raise PayloadTooLargeError(
            msg,
            details={"max_storage_bytes_per_principal": str(limit)},
        )

    async def _limited_stream() -> AsyncIterator[bytes]:
        seen = 0
        async for chunk in stream:
            seen += len(chunk)
            if used + seen > limit:
                exceeded_msg = (
                    f"Upload exceeds storage quota of {limit} bytes for principal "
                    f"{principal.principal_id!r}."
                )
                raise PayloadTooLargeError(
                    exceeded_msg,
                    details={"max_storage_bytes_per_principal": str(limit)},
                )
            yield chunk

    return _limited_stream()
