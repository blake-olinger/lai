"""Shared unit and integration test helpers (explicit imports for strict type checking)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

from lai.config.settings import LaiSettings
from lai.domain.guardrails import GuardrailConfig
from lai.domain.observability import LoggingConfig
from lai.ports.orchestrator import OrchestratorPort

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator
    from pathlib import Path

    from fastapi import FastAPI

    from lai.domain.chat import OrchestratorEvent, TurnOutput


class FailingEventStream:
    """AsyncIterator[OrchestratorEvent] raising a configured exception on first iteration."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __aiter__(self) -> FailingEventStream:
        return self

    async def __anext__(self) -> OrchestratorEvent:
        raise self._exc


class StubOrchestratorBase(OrchestratorPort):
    """Base orchestrator double providing unimplemented client-tool resume stubs."""

    async def resume_client_tool(self, *_: object, **__: object) -> TurnOutput:
        raise NotImplementedError

    def resume_client_tool_stream(
        self, *_: object, **__: object
    ) -> AsyncIterator[OrchestratorEvent]:
        return FailingEventStream(NotImplementedError())


def bearer_headers(token: str, **extra: str) -> dict[str, str]:
    """Build Authorization Bearer headers merged with optional extra headers."""
    return {"Authorization": f"Bearer {token}", **extra}


def build_base_settings(
    tmp_path: Path,
    *,
    enable_file_logs: bool = False,
    guardrails: GuardrailConfig | None = None,
) -> LaiSettings:
    """Construct isolated LaiSettings rooted under tmp_path."""
    data_dir = tmp_path / "data"
    return LaiSettings(
        data_dir=data_dir,
        models_dir=tmp_path / "models",
        bin_dir=tmp_path / "bin",
        logging=LoggingConfig(log_dir=data_dir / "logs", enable_file_logs=enable_file_logs),
        guardrails=guardrails or GuardrailConfig(enabled=False),
    )


@asynccontextmanager
async def open_app_client(
    app: FastAPI,
    *,
    raise_app_exceptions: bool = True,
) -> AsyncGenerator[httpx.AsyncClient]:
    """Enter FastAPI lifespan and yield an httpx.AsyncClient bound via ASGITransport."""
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        yield client
