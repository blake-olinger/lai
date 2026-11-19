"""FastAPI application factory assembling middleware, error handlers, routes, and lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from lai.api._cors import (
    DEFAULT_CORS_ALLOW_HEADERS,
    DynamicCorsController,
    DynamicCorsMiddleware,
)
from lai.api.errors import register_exception_handlers
from lai.api.middleware import RequestContextMiddleware
from lai.api.routes import api_router
from lai.api.runtime import (
    build_app_services,
    cancel_background_tasks,
    close_app_services,
)
from lai.config.settings import LaiSettings, load_settings
from lai.domain.version import PACKAGE_VERSION

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from lai.api.deps import AppServices
    from lai.ports.orchestrator import OrchestratorPort

APP_TITLE = "LAI Local AI Server"
APP_VERSION = PACKAGE_VERSION
__all__ = ["APP_TITLE", "APP_VERSION", "DEFAULT_CORS_ALLOW_HEADERS", "create_app"]


def create_app(
    settings: LaiSettings | None = None,
    *,
    services: AppServices | None = None,
    orchestrator: OrchestratorPort | None = None,
    config_path: Path | None = None,
) -> FastAPI:
    """Construct the FastAPI application with optional prebuilt services or orchestrator."""
    resolved_settings = settings or (services.settings if services is not None else load_settings())
    cors_controller = (
        services.cors_controller
        if services is not None and services.cors_controller is not None
        else DynamicCorsController(
            resolved_settings.cors_allowed_origins,
            resolved_settings.cors_extra_allow_headers,
        )
    )
    if services is not None:
        services.cors_controller = cors_controller
        cors_controller.reconfigure(
            services.settings.cors_allowed_origins,
            services.settings.cors_extra_allow_headers,
        )

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
            config_path=config_path,
        )
        built_services.cors_controller = cors_controller
        cors_controller.reconfigure(
            built_services.settings.cors_allowed_origins,
            built_services.settings.cors_extra_allow_headers,
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

    app.add_middleware(DynamicCorsMiddleware, controller=cors_controller)
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app
