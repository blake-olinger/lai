"""FastAPI API route package aggregating all domain routers."""

from __future__ import annotations

from fastapi import APIRouter

from lai.api.routes.files import router as files_router
from lai.api.routes.health import router as health_router
from lai.api.routes.openai import router as openai_router
from lai.api.routes.projects import router as projects_router
from lai.api.routes.sessions import router as sessions_router
from lai.api.routes.tools import router as tools_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(projects_router)
api_router.include_router(files_router)
api_router.include_router(sessions_router)
api_router.include_router(openai_router)
api_router.include_router(tools_router)

__all__ = [
    "api_router",
    "files_router",
    "health_router",
    "openai_router",
    "projects_router",
    "sessions_router",
    "tools_router",
]
