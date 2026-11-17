"""Project API request and response schemas."""

from __future__ import annotations

from pydantic import Field

from lai.domain.base import StrictModel
from lai.domain.projects import Project


class ProjectCreateRequest(StrictModel):
    """Payload for creating a new isolated workspace project."""

    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    workspace_root: str | None = None
    default_model_profile: str = "gemma4-26b-a4b"


class ProjectListResponse(StrictModel):
    """Paginated list of workspace projects."""

    items: list[Project]
    count: int = Field(ge=0)
