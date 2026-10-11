from typing import Any, Protocol

from ..models import Job


class Backend(Protocol):
    async def submit(self, text: str, idem_key: str | None) -> tuple[Job, bool]:
        """Store a job and enqueue it. Returns (job, created); created is False for a repeated key."""

    async def get(self, job_id: str) -> Job | None: ...

    async def take_batch(self, max_size: int, max_wait_s: float) -> list[Job]:
        """Block for a first job, then keep filling until max_size or max_wait_s has passed."""

    async def complete(self, job: Job, result: dict[str, Any]) -> None: ...

    async def fail(self, job: Job, error: str, max_attempts: int) -> bool:
        """Requeue the job or move it to the dead-letter queue. Returns True if it was requeued."""

    async def depths(self) -> tuple[int, int]:
        """(queued, dead)"""

    async def dead_letters(self, limit: int = 50) -> list[Job]: ...

    async def close(self) -> None: ...
