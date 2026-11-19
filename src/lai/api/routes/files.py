"""HTTP-only multipart and raw streaming file upload, chunk listing, and deletion routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)

from lai.api._tenancy import (
    enforce_and_wrap_storage_quota,
    require_accessible_file,
    require_accessible_project,
)
from lai.api.deps import AppServices, get_services, require_scope_dep
from lai.api.schemas.common import DeleteResponse
from lai.api.schemas.files import FileChunkListResponse, FileListResponse
from lai.domain.auth import ApiScope, AuthPrincipal
from lai.domain.errors import ValidationFailedError
from lai.domain.files import FileRecord, FileStatus
from lai.domain.ids import new_file_id
from lai.ingestion.detector import detect_file_kind

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

router = APIRouter(prefix="/v1/files", tags=["files"])

_UPLOAD_CHUNK_SIZE = 65_536


@dataclass(slots=True)
class _UploadPayload:
    """Normalized incoming upload descriptor passed to ingestion/background storage."""

    project_id: str
    filename: str
    content_type: str
    stream: AsyncIterator[bytes]
    wait: bool


async def _iter_upload_file(upload: UploadFile) -> AsyncIterator[bytes]:
    """Yield chunks from a FastAPI UploadFile without loading the entire file into RAM."""
    while True:
        chunk = await upload.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


async def _ingest_or_schedule(
    services: AppServices,
    response: Response,
    payload: _UploadPayload,
    *,
    principal: AuthPrincipal,
) -> FileRecord:
    """Verify project access + storage quota and ingest synchronously (201) or async (202)."""
    await require_accessible_project(services, payload.project_id, principal)
    quota_stream = await enforce_and_wrap_storage_quota(services, principal, payload.stream)

    if payload.wait:
        response.status_code = status.HTTP_201_CREATED
        return await services.ingestion_pipeline.ingest_upload(
            payload.project_id,
            payload.filename,
            payload.content_type,
            quota_stream,
        )

    file_id = new_file_id()
    blob_path, size_bytes, sha256 = await services.blob_store.write_stream(
        payload.project_id,
        file_id,
        payload.filename,
        quota_stream,
    )
    pending = FileRecord(
        id=file_id,
        project_id=payload.project_id,
        filename=payload.filename,
        content_type=payload.content_type,
        size_bytes=size_bytes,
        sha256=sha256,
        blob_path=str(blob_path),
        kind=detect_file_kind(payload.filename, payload.content_type, ""),
        status=FileStatus.PENDING,
    )
    persisted = await services.file_repo.upsert_file(pending)
    services.schedule_background_task(
        services.ingestion_pipeline.reindex_file(persisted.id),
        name=f"ingest-{persisted.id}",
    )
    response.status_code = status.HTTP_202_ACCEPTED
    return persisted


@router.post(
    "",
    response_model=FileRecord,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_202_ACCEPTED: {
            "model": FileRecord,
            "description": "Accepted for asynchronous background indexing when wait=false.",
        },
    },
)
async def upload_file_multipart(  # noqa: PLR0913, PLR0917
    response: Response,
    file: Annotated[UploadFile, File(...)],
    project_id: Annotated[str, Form(min_length=1)],
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.FILES_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
    wait: Annotated[bool, Query()] = True,
    wait_form: Annotated[bool | None, Form(alias="wait")] = None,
) -> FileRecord:
    """Upload a project file via multipart/form-data and ingest synchronously or asynchronously."""
    effective_wait = wait_form if wait_form is not None else wait
    filename = (file.filename or "").strip()
    if not filename:
        msg = "Uploaded multipart file must include a non-empty filename."
        raise ValidationFailedError(msg, details={"field": "file"})
    content_type = file.content_type or "application/octet-stream"
    return await _ingest_or_schedule(
        services,
        response,
        _UploadPayload(
            project_id=project_id,
            filename=filename,
            content_type=content_type,
            stream=_iter_upload_file(file),
            wait=effective_wait,
        ),
        principal=principal,
    )


@router.post(
    "/stream",
    response_model=FileRecord,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_202_ACCEPTED: {
            "model": FileRecord,
            "description": "Accepted for asynchronous background indexing when wait=false.",
        },
    },
)
async def upload_file_stream(  # noqa: PLR0913, PLR0917
    request: Request,
    response: Response,
    x_lai_filename: Annotated[str, Header(alias="X-LAI-Filename", min_length=1)],
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.FILES_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
    project_id: Annotated[str | None, Query(min_length=1)] = None,
    x_lai_project_id: Annotated[str | None, Header(alias="X-LAI-Project-Id")] = None,
    wait: Annotated[bool, Query()] = True,
) -> FileRecord:
    """Stream a raw HTTP request body directly into project blob storage and ingestion."""
    resolved_project_id = (project_id or x_lai_project_id or "").strip()
    if not resolved_project_id:
        msg = "project_id query parameter or X-LAI-Project-Id header is required."
        raise ValidationFailedError(msg, details={"field": "project_id"})
    filename = x_lai_filename.strip()
    if not filename:
        msg = "X-LAI-Filename header must not be blank."
        raise ValidationFailedError(msg, details={"field": "X-LAI-Filename"})

    content_type = request.headers.get("content-type", "application/octet-stream")
    return await _ingest_or_schedule(
        services,
        response,
        _UploadPayload(
            project_id=resolved_project_id,
            filename=filename,
            content_type=content_type,
            stream=request.stream(),
            wait=wait,
        ),
        principal=principal,
    )


@router.get("", response_model=FileListResponse)
async def list_files(
    project_id: Annotated[str, Query(min_length=1)],
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.FILES_READ))],
    services: Annotated[AppServices, Depends(get_services)],
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> FileListResponse:
    """List uploaded files scoped to an accessible project."""
    await require_accessible_project(services, project_id, principal)
    items = await services.file_repo.list_files(project_id, limit=limit, offset=offset)
    return FileListResponse(items=items, count=len(items))


@router.get("/{file_id}", response_model=FileRecord)
async def get_file(
    file_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.FILES_READ))],
    services: Annotated[AppServices, Depends(get_services)],
) -> FileRecord:
    """Fetch metadata and deterministic summaries for an accessible uploaded file."""
    return await require_accessible_file(services, file_id, principal)


@router.get("/{file_id}/chunks", response_model=FileChunkListResponse)
async def list_file_chunks(
    file_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.FILES_READ))],
    services: Annotated[AppServices, Depends(get_services)],
    start_line: Annotated[int | None, Query(ge=1)] = None,
    end_line: Annotated[int | None, Query(ge=1)] = None,
) -> FileChunkListResponse:
    """Return ordered indexed chunks for file_id, optionally filtered by line range."""
    if start_line is not None and end_line is not None and end_line < start_line:
        msg = f"end_line ({end_line}) must be >= start_line ({start_line})."
        raise ValidationFailedError(msg, details={"field": "end_line"})
    await require_accessible_file(services, file_id, principal)
    chunks = await services.file_repo.list_chunks(
        file_id,
        start_line=start_line,
        end_line=end_line,
    )
    return FileChunkListResponse(file_id=file_id, items=chunks, count=len(chunks))


@router.delete("/{file_id}", response_model=DeleteResponse)
async def delete_file(
    file_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_scope_dep(ApiScope.FILES_WRITE))],
    services: Annotated[AppServices, Depends(get_services)],
) -> DeleteResponse:
    """Delete a file's blob, SQLite metadata/chunks/FTS5 rows, and LanceDB vectors."""
    record = await require_accessible_file(services, file_id, principal)
    if services.vector_store is not None:
        await services.vector_store.delete_file_vectors(record.project_id, record.id)
    deleted = await services.file_repo.delete_file(record.id)
    await services.blob_store.delete_blob(Path(record.blob_path))
    return DeleteResponse(id=file_id, deleted=deleted)
