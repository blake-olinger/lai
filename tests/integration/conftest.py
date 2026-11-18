"""Shared pytest fixtures for ASGI integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._helpers import build_base_settings, open_app_client
from tests.integration._helpers import (
    IntegrationHarness,
    IntegrationSpec,
    open_services_harness,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    import httpx

    from lai.config.settings import LaiSettings


@pytest.fixture
def integration_settings(tmp_path: Path) -> LaiSettings:
    """Return isolated LaiSettings with guardrails and file logs disabled."""
    return build_base_settings(tmp_path)


@pytest.fixture
def services_harness_factory() -> Callable[
    [LaiSettings, IntegrationSpec | None],
    AbstractAsyncContextManager[IntegrationHarness],
]:
    """Return the shared real AppServices + httpx.AsyncClient context manager."""
    return open_services_harness


@pytest.fixture
def asgi_client_factory() -> Callable[..., AbstractAsyncContextManager[httpx.AsyncClient]]:
    """Return the shared FastAPI lifespan + httpx.AsyncClient context manager."""
    return open_app_client
