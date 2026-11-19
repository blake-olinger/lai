"""Project domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from lai.domain.base import StrictModel, utc_now
from lai.domain.ids import new_project_id


class Project(StrictModel):
    """Isolated workspace project containing files, sessions, and indexes."""

    id: str = Field(default_factory=new_project_id)
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    default_model_profile: str = "gemma4-26b-a4b"
    owner_principal_id: str = Field(default="system", min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CreateProjectInput(StrictModel):
    """Input payload for creating a new project."""

    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    default_model_profile: str = "gemma4-26b-a4b"
