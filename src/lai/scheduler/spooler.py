"""Single-slot priority inference spooler enforcing exclusive GPU leases."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from lai.domain.errors import QueueFullError, ValidationFailedError
from lai.domain.observability import AlertCategory
from lai.domain.scheduler import LeaseRequest, SpoolerSnapshot
from lai.observability import bind_log_context, get_component_logger
from lai.ports.scheduler import InferenceSpoolerPort
from lai.scheduler.fair_queue import FairPriorityQueue

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable


class InferenceSpooler(InferenceSpoolerPort):
    """Single-slot (max_inflight == 1) fair priority spooler for GPU inference."""

    def __init__(
        self,
        max_queued_jobs: int = 64,
        *,
        max_queued_jobs_per_principal: int | None = None,
        acquire_timeout_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if acquire_timeout_seconds is not None and acquire_timeout_seconds <= 0:
            msg = "acquire_timeout_seconds must be positive when set."
            raise ValidationFailedError(msg)
        self._queue = FairPriorityQueue(
            max_queued_jobs=max_queued_jobs,
            max_queued_jobs_per_principal=max_queued_jobs_per_principal,
        )
        self._acquire_timeout_seconds = acquire_timeout_seconds
        self._clock = clock
        self._logger = get_component_logger("scheduler")

        self._active_request: LeaseRequest | None = None
        self._waiters: dict[str, asyncio.Future[None]] = {}
        self._total_completed = 0
        self._total_rejected = 0
        self._total_evicted = 0
        self._total_wait_ms = 0.0
        self._wait_samples = 0

    @asynccontextmanager
    async def lease(
        self,
        request: LeaseRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncGenerator[None]:
        """Acquire the exclusive GPU inference slot according to priority and fairness."""
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self._acquire_timeout_seconds
        )
        if effective_timeout is not None and effective_timeout <= 0:
            msg = "timeout_seconds must be positive when set."
            raise ValidationFailedError(msg)

        with bind_log_context(
            job_id=request.job_id,
            session_id=request.session_id,
            project_id=request.project_id,
            principal_id=request.principal_id,
        ):
            await self._acquire_slot(request, effective_timeout=effective_timeout)
            try:
                yield
                self._total_completed += 1
            finally:
                self._release_slot(request.job_id)

    def snapshot(self) -> SpoolerSnapshot:
        """Return a point-in-time snapshot of active, queued, and cumulative spooler stats."""
        avg_wait_ms = self._total_wait_ms / self._wait_samples if self._wait_samples > 0 else 0.0
        active = self._active_request
        return SpoolerSnapshot(
            active_job_id=active.job_id if active is not None else None,
            active_session_id=active.session_id if active is not None else None,
            active_priority=active.priority if active is not None else None,
            queued_count=len(self._queue),
            max_queued_jobs=self._queue.max_queued_jobs,
            queued_by_priority=self._queue.counts_by_priority(),
            total_completed=self._total_completed,
            total_rejected=self._total_rejected,
            total_evicted=self._total_evicted,
            avg_wait_ms=avg_wait_ms,
        )

    async def _acquire_slot(
        self,
        request: LeaseRequest,
        *,
        effective_timeout: float | None,
    ) -> None:
        if self._active_request is not None and self._active_request.job_id == request.job_id:
            msg = f"Job {request.job_id!r} already holds the active lease."
            raise ValidationFailedError(msg, details={"job_id": request.job_id})

        if self._active_request is None:
            self._active_request = request
            self._record_wait(0.0)
            self._logger.info(
                "spooler_lease_acquired",
                priority=request.priority.name,
                wait_ms=0.0,
            )
            return

        entered_at = self._clock()
        self._enqueue_or_reject(request)
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        self._waiters[request.job_id] = waiter

        await self._await_waiter(
            request,
            waiter,
            entered_at=entered_at,
            effective_timeout=effective_timeout,
        )
        wait_ms = self._elapsed_ms(entered_at)
        self._record_wait(wait_ms)
        self._logger.info(
            "spooler_lease_acquired",
            priority=request.priority.name,
            wait_ms=round(wait_ms, 2),
        )

    def _enqueue_or_reject(self, request: LeaseRequest) -> None:
        try:
            self._queue.enqueue(request)
        except QueueFullError:
            self._total_rejected += 1
            self._logger.warning(
                "spooler_queue_full",
                priority=request.priority.name,
                queued_count=len(self._queue),
                max_queued_jobs=self._queue.max_queued_jobs,
            )
            raise
        self._logger.info(
            "spooler_enqueued",
            priority=request.priority.name,
            queued_count=len(self._queue),
        )

    async def _await_waiter(
        self,
        request: LeaseRequest,
        waiter: asyncio.Future[None],
        *,
        entered_at: float,
        effective_timeout: float | None,
    ) -> None:
        try:
            if effective_timeout is None:
                await waiter
            else:
                async with asyncio.timeout(effective_timeout):
                    await waiter
        except TimeoutError:
            wait_ms = self._elapsed_ms(entered_at)
            self._handle_wait_abort(request.job_id)
            self._logger.exception(
                "spooler_acquire_timeout",
                priority=request.priority.name,
                wait_ms=round(wait_ms, 2),
                timeout_seconds=effective_timeout,
                alert_category=AlertCategory.TIMEOUT.value,
            )
            raise
        except asyncio.CancelledError:
            wait_ms = self._elapsed_ms(entered_at)
            self._handle_wait_abort(request.job_id)
            self._logger.info(
                "spooler_evicted",
                priority=request.priority.name,
                wait_ms=round(wait_ms, 2),
                reason="cancelled",
            )
            raise

    def _handle_wait_abort(self, job_id: str) -> None:
        self._waiters.pop(job_id, None)
        self._queue.evict(job_id)
        self._total_evicted += 1
        if self._active_request is not None and self._active_request.job_id == job_id:
            self._release_slot(job_id)

    def _release_slot(self, job_id: str) -> None:
        if self._active_request is not None and self._active_request.job_id == job_id:
            released_priority = self._active_request.priority.name
            self._active_request = None
            self._logger.info("spooler_lease_released", priority=released_priority)
        self._dispatch_next()

    def _dispatch_next(self) -> None:
        while self._active_request is None:
            next_request = self._queue.dequeue()
            if next_request is None:
                return
            waiter = self._waiters.pop(next_request.job_id, None)
            if waiter is not None and not waiter.done():
                self._active_request = next_request
                waiter.set_result(None)
                return

    def _elapsed_ms(self, started_at: float) -> float:
        return max(0.0, (self._clock() - started_at) * 1000.0)

    def _record_wait(self, wait_ms: float) -> None:
        self._total_wait_ms += wait_ms
        self._wait_samples += 1
