"""Shared pytest fixtures across unit and integration tests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from tests._helpers import FailingEventStream, bearer_headers

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


@pytest.fixture(autouse=True)
def _teardown_root_logging() -> Generator[None]:
    """Flush, close, and clear root logger handlers after every test."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
    root.handlers.clear()


@pytest.fixture
def failing_stream_cls() -> type[FailingEventStream]:
    """Return the shared FailingEventStream async iterator class."""
    return FailingEventStream


@pytest.fixture
def auth_headers_factory() -> Callable[..., dict[str, str]]:
    """Return the shared Bearer authorization header builder."""
    return bearer_headers
