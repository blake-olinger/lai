"""Shared SSE priming and keepalive streaming helpers for chat and OpenAI endpoints."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lai.domain.chat import ContentDeltaEvent, KeepaliveEvent, OrchestratorEvent, TurnCompletedEvent
from lai.scheduler import stream_with_sse_keepalive

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

_DEFAULT_PEEK_BUDGET_SECONDS = 0.25


@dataclass(slots=True)
class PrimedSseStream:
    """Result of priming an SSE orchestrator stream before sending HTTP headers."""

    events: AsyncIterator[OrchestratorEvent]
    guardrail_blocked: bool
    guardrail_policy: str | None


def build_guardrail_headers(
    *,
    guardrail_blocked: bool,
    guardrail_policy: str | None,
) -> dict[str, str]:
    """Build X-LAI-Guardrail-* headers when a turn was blocked by guardrails."""
    if not guardrail_blocked:
        return {}
    headers = {"X-LAI-Guardrail-Blocked": "true"}
    if guardrail_policy:
        headers["X-LAI-Guardrail-Policy"] = guardrail_policy
    return headers


async def _cancel_pending(task: asyncio.Task[OrchestratorEvent] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, StopAsyncIteration):
        await task


async def _close_async_iter(iterator: AsyncIterator[OrchestratorEvent]) -> None:
    aclose = getattr(iterator, "aclose", None)
    if callable(aclose):
        result = aclose()
        if inspect.isawaitable(result):
            await result


async def _next_event(iterator: AsyncIterator[OrchestratorEvent]) -> OrchestratorEvent:
    """Await the next item from an OrchestratorEvent async iterator as a coroutine."""
    return await anext(iterator)


async def _consume_priming_window(
    guarded: AsyncIterator[OrchestratorEvent],
    *,
    budget_seconds: float,
) -> tuple[
    list[OrchestratorEvent],
    asyncio.Task[OrchestratorEvent] | None,
    bool,
    str | None,
]:
    """Peek initial immediate events without cancelling slow producer tasks."""
    buffered: list[OrchestratorEvent] = []
    pending_task: asyncio.Task[OrchestratorEvent] | None = None
    guardrail_blocked = False
    guardrail_policy: str | None = None

    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_seconds

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break

        task: asyncio.Task[OrchestratorEvent] = asyncio.create_task(_next_event(guarded))
        done, _ = await asyncio.wait({task}, timeout=remaining)
        if task not in done:
            pending_task = task
            break

        try:
            event = task.result()
        except StopAsyncIteration:
            break
        except BaseException:
            await _close_async_iter(guarded)
            raise

        buffered.append(event)
        if isinstance(event, TurnCompletedEvent):
            guardrail_blocked = event.output.guardrail_blocked
            guardrail_policy = event.output.guardrail_policy
            break
        if not isinstance(event, ContentDeltaEvent):
            break

    return buffered, pending_task, guardrail_blocked, guardrail_policy


async def prime_sse_stream(
    events: AsyncIterator[OrchestratorEvent],
    *,
    keepalive_interval_seconds: float,
    is_disconnected: Callable[[], Awaitable[bool]],
    peek_budget_seconds: float = _DEFAULT_PEEK_BUDGET_SECONDS,
) -> PrimedSseStream:
    """Wrap orchestrator stream in keepalives and inspect fast terminal guardrail rulings."""
    guarded = stream_with_sse_keepalive(
        events,
        keepalive_factory=KeepaliveEvent,
        interval_seconds=keepalive_interval_seconds,
        is_disconnected=is_disconnected,
    )
    budget = min(peek_budget_seconds, keepalive_interval_seconds * 0.5)
    buffered, pending_task, blocked, policy = await _consume_priming_window(
        guarded,
        budget_seconds=budget,
    )

    async def _replay() -> AsyncIterator[OrchestratorEvent]:
        nonlocal pending_task
        try:
            for item in buffered:
                yield item
            if pending_task is not None:
                task = pending_task
                pending_task = None
                try:
                    yield await task
                except StopAsyncIteration:
                    return
            async for item in guarded:
                yield item
        finally:
            await _cancel_pending(pending_task)
            await _close_async_iter(guarded)

    return PrimedSseStream(
        events=_replay(),
        guardrail_blocked=blocked,
        guardrail_policy=policy,
    )
