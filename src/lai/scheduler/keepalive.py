"""SSE and non-streaming JSON keepalive helpers preventing Cloudflare 524 timeouts."""

from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from lai.domain.errors import ClientDisconnectedError, ValidationFailedError

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable

    type DisconnectCheck = Callable[[], bool | Awaitable[bool]]

_KEEPALIVE_BYTE = b"\n"
_DISCONNECT_POLL_SECONDS = 0.05


class _StreamEnd:
    """Internal sentinel signaling producer task completion."""


class _WaitTimeout:
    """Internal sentinel signaling queue wait timeout."""


_STREAM_END = _StreamEnd()
_WAIT_TIMEOUT = _WaitTimeout()


@dataclass(frozen=True, slots=True)
class _SseStep[T]:
    """Single step emitted by the SSE keepalive consumer loop."""

    kind: Literal["item", "keepalive", "end", "disconnect"]
    event: T | None = None


async def _check_disconnected(is_disconnected: DisconnectCheck | None) -> bool:
    if is_disconnected is None:
        return False
    result = is_disconnected()
    if inspect.isawaitable(result):
        return bool(await result)
    return bool(result)


async def _cancel_task(task: asyncio.Task[object]) -> None:
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def _retrieve_task_exception(task: asyncio.Task[object]) -> None:
    if task.done() and not task.cancelled():
        _ = task.exception()


def _raise_if_failed(task: asyncio.Task[object]) -> None:
    if task.done() and not task.cancelled():
        exc = task.exception()
        if exc is not None:
            raise exc


def _compute_wait_step(
    remaining: float,
    is_disconnected: DisconnectCheck | None,
) -> float:
    if is_disconnected is not None:
        return min(remaining, _DISCONNECT_POLL_SECONDS)
    return remaining


async def _wait_for_queue_item[T](
    queue: asyncio.Queue[T | _StreamEnd],
    *,
    wait_seconds: float,
) -> T | _StreamEnd | _WaitTimeout:
    try:
        async with asyncio.timeout(wait_seconds):
            return await queue.get()
    except TimeoutError:
        return _WAIT_TIMEOUT


def _decode_pulled_item[T](
    pulled: T | _StreamEnd,
    producer_task: asyncio.Task[None],
) -> _SseStep[T]:
    if isinstance(pulled, _StreamEnd):
        _raise_if_failed(producer_task)
        return _SseStep("end")
    return _SseStep("item", pulled)


async def _pull_next_sse_event[T](
    queue: asyncio.Queue[T | _StreamEnd],
    producer_task: asyncio.Task[None],
    *,
    interval_seconds: float,
    is_disconnected: DisconnectCheck | None,
) -> _SseStep[T]:
    """Wait for the next SSE event, keepalive tick, stream end, or disconnect."""
    loop = asyncio.get_running_loop()
    next_keepalive = loop.time() + interval_seconds
    while True:
        if await _check_disconnected(is_disconnected):
            await _cancel_task(producer_task)
            return _SseStep("disconnect")
        if not queue.empty():
            return _decode_pulled_item(queue.get_nowait(), producer_task)

        remaining = next_keepalive - loop.time()
        if remaining <= 0:
            return _SseStep("keepalive")

        step = _compute_wait_step(remaining, is_disconnected)
        pulled = await _wait_for_queue_item(queue, wait_seconds=step)
        if not isinstance(pulled, _WaitTimeout):
            return _decode_pulled_item(pulled, producer_task)
        if loop.time() >= next_keepalive:
            return _SseStep("keepalive")


async def stream_with_sse_keepalive[T](
    events: AsyncIterable[T],
    *,
    keepalive_factory: Callable[[], T],
    interval_seconds: float = 10.0,
    is_disconnected: DisconnectCheck | None = None,
) -> AsyncIterator[T]:
    """Consume events eagerly, emit keepalives on idle intervals, and cancel on disconnect."""
    if interval_seconds <= 0:
        msg = "interval_seconds must be positive."
        raise ValidationFailedError(msg)

    queue: asyncio.Queue[T | _StreamEnd] = asyncio.Queue()

    async def _produce() -> None:
        async for event in events:
            await queue.put(event)

    producer_task = asyncio.create_task(_produce())
    producer_task.add_done_callback(lambda _t: queue.put_nowait(_STREAM_END))

    try:
        while True:
            step = await _pull_next_sse_event(
                queue,
                producer_task,
                interval_seconds=interval_seconds,
                is_disconnected=is_disconnected,
            )
            if step.kind in ("disconnect", "end"):
                return
            if step.kind == "keepalive":
                yield keepalive_factory()
            else:
                yield cast("T", step.event)
    finally:
        if not producer_task.done():
            await _cancel_task(producer_task)
        else:
            _retrieve_task_exception(producer_task)


def _json_default(obj: object) -> object:
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set | frozenset):
        typed_set = cast("set[object] | frozenset[object]", obj)
        return list(typed_set)
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


def _serialize_json_bytes(value: object) -> bytes:
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    model_dump_json = getattr(value, "model_dump_json", None)
    if callable(model_dump_json):
        dumped = model_dump_json()
        if isinstance(dumped, str):
            return dumped.encode("utf-8")
    return json.dumps(value, default=_json_default).encode("utf-8")


async def _await_task_step(
    task: asyncio.Task[object],
    *,
    wait_seconds: float,
) -> None:
    await asyncio.wait({task}, timeout=wait_seconds)


async def _keepalive_json_stream[T](
    task: asyncio.Task[T],
    *,
    interval_seconds: float,
    is_disconnected: DisconnectCheck | None,
    serialize: Callable[[T], bytes],
) -> AsyncIterator[bytes]:
    loop = asyncio.get_running_loop()
    try:
        if not task.done():
            if await _check_disconnected(is_disconnected):
                await _cancel_task(task)
                return
            yield _KEEPALIVE_BYTE

        while not task.done():
            next_keepalive = loop.time() + interval_seconds
            while not task.done():
                if await _check_disconnected(is_disconnected):
                    await _cancel_task(task)
                    return
                remaining = next_keepalive - loop.time()
                if remaining <= 0:
                    yield _KEEPALIVE_BYTE
                    break
                step = _compute_wait_step(remaining, is_disconnected)
                await _await_task_step(task, wait_seconds=step)

        result = await task
        yield serialize(result)
    finally:
        if not task.done():
            await _cancel_task(task)
        else:
            _retrieve_task_exception(task)


async def _to_task[T](awaitable: Awaitable[T]) -> T:
    return await awaitable


async def await_or_keepalive_json[T](
    coro: Awaitable[T],
    *,
    threshold_seconds: float = 20.0,
    interval_seconds: float = 15.0,
    is_disconnected: DisconnectCheck | None = None,
    serializer: Callable[[T], bytes] | None = None,
) -> T | AsyncIterator[bytes]:
    """Return resolved value within threshold or stream RFC 8259 whitespace before JSON bytes."""
    if threshold_seconds <= 0:
        msg = "threshold_seconds must be positive."
        raise ValidationFailedError(msg)
    if interval_seconds <= 0:
        msg = "interval_seconds must be positive."
        raise ValidationFailedError(msg)

    loop = asyncio.get_running_loop()
    task: asyncio.Task[T] = (
        asyncio.create_task(coro)
        if inspect.iscoroutine(coro)
        else asyncio.create_task(_to_task(coro))
    )
    deadline = loop.time() + threshold_seconds
    handing_off_stream = False

    try:
        while not task.done():
            if await _check_disconnected(is_disconnected):
                await _cancel_task(task)
                msg = "Client disconnected while waiting for JSON response."
                raise ClientDisconnectedError(msg)
            remaining = deadline - loop.time()
            if remaining <= 0:
                handing_off_stream = True
                return _keepalive_json_stream(
                    task,
                    interval_seconds=interval_seconds,
                    is_disconnected=is_disconnected,
                    serialize=serializer or _serialize_json_bytes,
                )
            step = _compute_wait_step(remaining, is_disconnected)
            await _await_task_step(task, wait_seconds=step)
        return await task
    finally:
        if not handing_off_stream and not task.done():
            await _cancel_task(task)
