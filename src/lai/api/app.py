"""FastAPI application factory assembling middleware, error handlers, routes, and lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lai.api.errors import register_exception_handlers
from lai.api.middleware import RequestContextMiddleware
from lai.api.routes import api_router
from lai.api.runtime import (
    build_app_services,
    cancel_background_tasks,
    close_app_services,
)
from lai.config.settings import LaiSettings, load_settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from lai.api.deps import AppServices
    from lai.ports.orchestrator import OrchestratorPort

APP_TITLE = "LAI Local AI Server"
APP_VERSION = "0.1.0"


def create_app(
    settings: LaiSettings | None = None,
    *,
    services: AppServices | None = None,
    orchestrator: OrchestratorPort | None = None,
) -> FastAPI:
    """Construct the FastAPI application with optional prebuilt services or orchestrator."""
    resolved_settings = settings or (services.settings if services is not None else load_settings())

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None]:
        if services is not None:
            fastapi_app.state.services = services
            try:
                yield
            finally:
                await cancel_background_tasks(services)
            return

        built_services = await build_app_services(
            resolved_settings,
            orchestrator=orchestrator,
        )
        fastapi_app.state.services = built_services
        try:
            yield
        finally:
            await close_app_services(built_services)

    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    if services is not None:
        app.state.services = services

    if resolved_settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "X-Request-Id",
                "X-LAI-Project-Id",
                "X-LAI-Session-Id",
                "X-LAI-File-Ids",
                "X-LAI-Filename",
                "X-LAI-Context-Mode",
            ],
            expose_headers=[
                "X-Request-Id",
                "Retry-After",
                "X-LAI-Guardrail-Blocked",
                "X-LAI-Guardrail-Policy",
            ],
        )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app
