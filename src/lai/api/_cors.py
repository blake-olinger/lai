"""Dynamic zero-restart ASGI CORS middleware and controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.middleware.cors import CORSMiddleware

if TYPE_CHECKING:
    from collections.abc import Sequence

    from starlette.types import ASGIApp, Receive, Scope, Send

DEFAULT_CORS_ALLOW_HEADERS: tuple[str, ...] = (
    "Authorization",
    "Content-Type",
    "X-Request-Id",
    "X-LAI-Project-Id",
    "X-LAI-Session-Id",
    "X-LAI-File-Ids",
    "X-LAI-Filename",
    "X-LAI-Context-Mode",
    # Official OpenAI JS/TS SDK (Stainless) browser headers
    "X-Stainless-Arch",
    "X-Stainless-Helper-Method",
    "X-Stainless-Lang",
    "X-Stainless-OS",
    "X-Stainless-Package-Version",
    "X-Stainless-Read-Timeout",
    "X-Stainless-Retry-Count",
    "X-Stainless-Runtime",
    "X-Stainless-Runtime-Version",
    "X-Stainless-Timeout",
)

CORS_ALLOW_METHODS: tuple[str, ...] = ("GET", "POST", "PATCH", "DELETE", "OPTIONS")
CORS_EXPOSE_HEADERS: tuple[str, ...] = (
    "X-Request-Id",
    "Retry-After",
    "X-LAI-Guardrail-Blocked",
    "X-LAI-Guardrail-Policy",
)


def merge_cors_allow_headers(extra_headers: Sequence[str]) -> list[str]:
    """Merge default CORS allow headers with optional custom headers case-insensitively."""
    merged: list[str] = list(DEFAULT_CORS_ALLOW_HEADERS)
    seen = {header.lower() for header in merged}
    for header in extra_headers:
        lowered = header.lower()
        if lowered not in seen:
            merged.append(header)
            seen.add(lowered)
    return merged


class DynamicCorsController:
    """Mutable holder allowing zero-restart CORS origin and header policy hot-swaps."""

    def __init__(
        self,
        origins: Sequence[str] = (),
        extra_allow_headers: Sequence[str] = (),
    ) -> None:
        self._origins: tuple[str, ...] = tuple(origins)
        self._extra_allow_headers: tuple[str, ...] = tuple(extra_allow_headers)
        self._revision = 0

    @property
    def revision(self) -> int:
        """Monotonic counter incremented whenever CORS policy changes."""
        return self._revision

    @property
    def origins(self) -> tuple[str, ...]:
        """Return currently allowed explicit CORS origins."""
        return self._origins

    @property
    def extra_allow_headers(self) -> tuple[str, ...]:
        """Return extra custom CORS request headers currently permitted."""
        return self._extra_allow_headers

    @property
    def effective_allow_headers(self) -> list[str]:
        """Return merged default and custom CORS allow headers."""
        return merge_cors_allow_headers(self._extra_allow_headers)

    def reconfigure(
        self,
        origins: Sequence[str],
        extra_allow_headers: Sequence[str],
    ) -> None:
        """Atomically update allowed CORS origins and extra headers for subsequent requests."""
        self._origins = tuple(origins)
        self._extra_allow_headers = tuple(extra_allow_headers)
        self._revision += 1

    def build_middleware(self, app: ASGIApp) -> CORSMiddleware | None:
        """Build a Starlette CORSMiddleware for current policy, or None when origins is empty."""
        if not self._origins:
            return None
        return CORSMiddleware(
            app=app,
            allow_origins=list(self._origins),
            allow_credentials=False,
            allow_methods=list(CORS_ALLOW_METHODS),
            allow_headers=self.effective_allow_headers,
            expose_headers=list(CORS_EXPOSE_HEADERS),
        )


class DynamicCorsMiddleware:
    """Pure ASGI wrapper rebuilding CORSMiddleware only when controller revision changes."""

    def __init__(self, app: ASGIApp, *, controller: DynamicCorsController) -> None:
        self.app = app
        self._controller = controller
        self._cached_revision = -1
        self._delegate: CORSMiddleware | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Dispatch HTTP requests through the latest live CORS policy without restarting."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        current_rev = self._controller.revision
        if current_rev != self._cached_revision:
            self._delegate = self._controller.build_middleware(self.app)
            self._cached_revision = current_rev

        if self._delegate is not None:
            await self._delegate(scope, receive, send)
            return
        await self.app(scope, receive, send)
