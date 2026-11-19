"""Priority-banded two-level fair round-robin queue (principal -> session lanes)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lai.domain.errors import QueueFullError, ValidationFailedError
from lai.domain.scheduler import JobPriority

if TYPE_CHECKING:
    from lai.domain.scheduler import LeaseRequest


@dataclass(slots=True)
class _PrincipalLanes:
    """Round-robin queue of per-session_id lanes belonging to a single principal."""

    session_order: deque[str] = field(default_factory=deque)
    sessions: dict[str, deque[LeaseRequest]] = field(default_factory=dict)

    def push(self, request: LeaseRequest) -> None:
        """Append request to its session lane within this principal."""
        session_queue = self.sessions.get(request.session_id)
        if session_queue is None:
            new_queue: deque[LeaseRequest] = deque()
            self.sessions[request.session_id] = new_queue
            self.session_order.append(request.session_id)
            session_queue = new_queue
        session_queue.append(request)

    def pop(self) -> LeaseRequest | None:
        """Pop the next request from the front session lane and rotate the session lane."""
        if not self.session_order:
            return None
        session_id = self.session_order.popleft()
        session_queue = self.sessions[session_id]
        request = session_queue.popleft()
        if session_queue:
            self.session_order.append(session_id)
        else:
            del self.sessions[session_id]
        return request

    def remove(self, request: LeaseRequest) -> bool:
        """Remove a specific waiting request from its session lane and prune if empty."""
        session_queue = self.sessions.get(request.session_id)
        if session_queue is None:
            return False
        for idx, item in enumerate(session_queue):
            if item.job_id == request.job_id:
                del session_queue[idx]
                if not session_queue:
                    del self.sessions[request.session_id]
                    self.session_order.remove(request.session_id)
                return True
        return False

    def is_empty(self) -> bool:
        """Return True when this principal has no waiting session lanes."""
        return not self.session_order


@dataclass(slots=True)
class _PriorityBand:
    """Two-level round-robin queue (outer principal_id -> inner session_id lanes)."""

    principal_order: deque[str] = field(default_factory=deque)
    principals: dict[str, _PrincipalLanes] = field(default_factory=dict)
    count: int = 0

    def push(self, request: LeaseRequest) -> None:
        """Append request to its (principal_id -> session_id) lane."""
        lanes = self.principals.get(request.principal_id)
        if lanes is None:
            lanes = _PrincipalLanes()
            self.principals[request.principal_id] = lanes
            self.principal_order.append(request.principal_id)
        lanes.push(request)
        self.count += 1

    def pop(self) -> LeaseRequest | None:
        """Pop one request from the front principal's next session lane and rotate both levels."""
        if not self.principal_order:
            return None
        principal_id = self.principal_order.popleft()
        lanes = self.principals[principal_id]
        request = lanes.pop()
        if request is None:
            del self.principals[principal_id]
            return None
        self.count -= 1
        if not lanes.is_empty():
            self.principal_order.append(principal_id)
        else:
            del self.principals[principal_id]
        return request

    def remove(self, request: LeaseRequest) -> bool:
        """Remove a specific waiting request and prune empty principal/session entries."""
        lanes = self.principals.get(request.principal_id)
        if lanes is None or not lanes.remove(request):
            return False
        self.count -= 1
        if lanes.is_empty():
            del self.principals[request.principal_id]
            self.principal_order.remove(request.principal_id)
        return True


class FairPriorityQueue:
    """Priority-banded queue scheduling hierarchical round-robin across principals and sessions."""

    def __init__(
        self,
        max_queued_jobs: int = 64,
        max_queued_jobs_per_principal: int | None = None,
    ) -> None:
        if max_queued_jobs < 1:
            msg = "max_queued_jobs must be at least 1."
            raise ValidationFailedError(msg)
        if max_queued_jobs_per_principal is not None and max_queued_jobs_per_principal < 1:
            msg = "max_queued_jobs_per_principal must be at least 1 when set."
            raise ValidationFailedError(msg)
        self._max_queued_jobs = max_queued_jobs
        self._max_queued_jobs_per_principal = max_queued_jobs_per_principal
        self._bands: dict[JobPriority, _PriorityBand] = {
            priority: _PriorityBand() for priority in sorted(JobPriority)
        }
        self._by_job_id: dict[str, LeaseRequest] = {}
        self._queued_by_principal: dict[str, int] = {}

    @property
    def max_queued_jobs(self) -> int:
        """Return the maximum number of waiting jobs permitted in the queue."""
        return self._max_queued_jobs

    @property
    def max_queued_jobs_per_principal(self) -> int | None:
        """Return the optional per-principal waiting job quota."""
        return self._max_queued_jobs_per_principal

    def __len__(self) -> int:
        """Return the total number of waiting jobs across all priority bands."""
        return len(self._by_job_id)

    def __contains__(self, job_id: object) -> bool:
        """Return True if job_id is currently queued."""
        return isinstance(job_id, str) and job_id in self._by_job_id

    def is_empty(self) -> bool:
        """Return True when no jobs are waiting in any priority band."""
        return not self._by_job_id

    def queued_count_for_principal(self, principal_id: str) -> int:
        """Return the number of waiting jobs currently queued for a principal."""
        return self._queued_by_principal.get(principal_id, 0)

    def enqueue(self, request: LeaseRequest) -> None:
        """Enqueue request into its priority band or raise QueueFullError at capacity."""
        if request.job_id in self._by_job_id:
            msg = f"Job {request.job_id!r} is already queued."
            raise ValidationFailedError(msg, details={"job_id": request.job_id})
        if len(self._by_job_id) >= self._max_queued_jobs:
            msg = f"Inference queue is full (max_queued_jobs={self._max_queued_jobs})."
            raise QueueFullError(
                msg,
                details={"max_queued_jobs": str(self._max_queued_jobs)},
            )
        if (
            not request.bypass_principal_quota
            and self._max_queued_jobs_per_principal is not None
            and self._queued_by_principal.get(request.principal_id, 0)
            >= self._max_queued_jobs_per_principal
        ):
            msg = (
                f"Principal {request.principal_id!r} exceeded queued job quota "
                f"(max_queued_jobs_per_principal={self._max_queued_jobs_per_principal})."
            )
            raise QueueFullError(
                msg,
                details={
                    "principal_id": request.principal_id,
                    "max_queued_jobs_per_principal": str(self._max_queued_jobs_per_principal),
                },
            )
        self._bands[request.priority].push(request)
        self._by_job_id[request.job_id] = request
        self._queued_by_principal[request.principal_id] = (
            self._queued_by_principal.get(request.principal_id, 0) + 1
        )

    def dequeue(self) -> LeaseRequest | None:
        """Pop the highest-priority next request using hierarchical round-robin scheduling."""
        for band in self._bands.values():
            if band.count > 0:
                request = band.pop()
                if request is not None:
                    self._by_job_id.pop(request.job_id, None)
                    self._decrement_principal_count(request.principal_id)
                    return request
        return None

    def evict(self, job_id: str) -> bool:
        """Evict a waiting request by job_id, returning True if it was present."""
        request = self._by_job_id.pop(job_id, None)
        if request is None:
            return False
        removed = self._bands[request.priority].remove(request)
        if removed:
            self._decrement_principal_count(request.principal_id)
        return removed

    def counts_by_priority(self) -> dict[str, int]:
        """Return queued job counts keyed by JobPriority name."""
        return {priority.name: band.count for priority, band in self._bands.items()}

    def _decrement_principal_count(self, principal_id: str) -> None:
        current = self._queued_by_principal.get(principal_id, 0)
        if current <= 1:
            self._queued_by_principal.pop(principal_id, None)
        else:
            self._queued_by_principal[principal_id] = current - 1
