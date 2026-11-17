"""Uploaded file and chunk listing API schemas."""

from __future__ import annotations

from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.files import FileChunk, FileRecord


class FileListResponse(StrictModel):
    """List of uploaded files within a project."""

    items: list[FileRecord]
    count: int = Field(ge=0)


class FileChunkListResponse(StrictModel):
    """Ordered list of indexed chunks for a specific project file."""

    file_id: str = Field(min_length=1)
    items: list[FileChunk]
    count: int = Field(ge=0)
