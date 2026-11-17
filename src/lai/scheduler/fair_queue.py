"""Priority-banded fair round-robin queue across (principal_id, session_id) lanes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lai.domain.errors import QueueFullError, ValidationFailedError
from lai.domain.scheduler import JobPriority

if TYPE_CHECKING:
    from lai.domain.scheduler import LeaseRequest

LaneKey = tuple[str, str]


@dataclass(slots=True)
class _PriorityBand:
    """Round-robin queue of per-(principal_id, session_id) lanes for a single priority."""

    lane_order: deque[LaneKey] = field(default_factory=deque)
    lanes: dict[LaneKey, deque[LeaseRequest]] = field(default_factory=dict)
    count: int = 0

    def push(self, request: LeaseRequest) -> None:
        """Append request to its (principal_id, session_id) lane."""
        lane_key: LaneKey = (request.principal_id, request.session_id)
        lane = self.lanes.get(lane_key)
        if lane is None:
            new_lane: deque[LeaseRequest] = deque()
            self.lanes[lane_key] = new_lane
            self.lane_order.append(lane_key)
            lane = new_lane
        lane.append(request)
        self.count += 1

    def pop(self) -> LeaseRequest | None:
        """Pop the next request from the front lane and rotate the lane to the back."""
        if not self.lane_order:
            return None
        lane_key = self.lane_order.popleft()
        lane = self.lanes[lane_key]
        request = lane.popleft()
        self.count -= 1
        if lane:
            self.lane_order.append(lane_key)
        else:
            del self.lanes[lane_key]
        return request

    def remove(self, request: LeaseRequest) -> bool:
        """Remove a specific waiting request and prune its lane if empty."""
        lane_key: LaneKey = (request.principal_id, request.session_id)
        lane = self.lanes.get(lane_key)
        if lane is None:
            return False
        for idx, item in enumerate(lane):
            if item.job_id == request.job_id:
                del lane[idx]
                self.count -= 1
                if not lane:
                    del self.lanes[lane_key]
                    self.lane_order.remove(lane_key)
                return True
        return False


class FairPriorityQueue:
    """Priority-banded queue scheduling fair round-robin across (principal, session) lanes."""

    def __init__(self, max_queued_jobs: int = 64) -> None:
        if max_queued_jobs < 1:
            msg = "max_queued_jobs must be at least 1."
            raise ValidationFailedError(msg)
        self._max_queued_jobs = max_queued_jobs
        self._bands: dict[JobPriority, _PriorityBand] = {
            priority: _PriorityBand() for priority in sorted(JobPriority)
        }
        self._by_job_id: dict[str, LeaseRequest] = {}

    @property
    def max_queued_jobs(self) -> int:
        """Return the maximum number of waiting jobs permitted in the queue."""
        return self._max_queued_jobs

    def __len__(self) -> int:
        """Return the total number of waiting jobs across all priority bands."""
        return len(self._by_job_id)

    def __contains__(self, job_id: object) -> bool:
        """Return True if job_id is currently queued."""
        return isinstance(job_id, str) and job_id in self._by_job_id

    def is_empty(self) -> bool:
        """Return True when no jobs are waiting in any priority band."""
        return not self._by_job_id

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
        self._bands[request.priority].push(request)
        self._by_job_id[request.job_id] = request

    def dequeue(self) -> LeaseRequest | None:
        """Pop the highest-priority next request using per-band round-robin scheduling."""
        for band in self._bands.values():
            if band.count > 0:
                request = band.pop()
                if request is not None:
                    self._by_job_id.pop(request.job_id, None)
                    return request
        return None

    def evict(self, job_id: str) -> bool:
        """Evict a waiting request by job_id, returning True if it was present."""
        request = self._by_job_id.pop(job_id, None)
        if request is None:
            return False
        return self._bands[request.priority].remove(request)

    def counts_by_priority(self) -> dict[str, int]:
        """Return queued job counts keyed by JobPriority name."""
        return {priority.name: band.count for priority, band in self._bands.items()}
