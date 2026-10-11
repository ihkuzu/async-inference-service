import asyncio
import time
from collections import deque
from typing import Any

from ..models import Job, Status


class MemoryBackend:
    """Single-process backend for tests and local runs."""

    def __init__(self, idem_ttl_s: int = 86400, **_):
        self.jobs: dict[str, Job] = {}
        self.idem: dict[str, tuple[str, float]] = {}
        self.queue: deque[str] = deque()
        self.dead: list[str] = []
        self.idem_ttl = idem_ttl_s
        self.cond = asyncio.Condition()

    async def submit(self, text, idem_key):
        async with self.cond:
            if idem_key:
                hit = self.idem.get(idem_key)
                if hit and hit[1] > time.time():
                    return self.jobs[hit[0]], False
            job = Job(text=text)
            self.jobs[job.id] = job
            if idem_key:
                self.idem[idem_key] = (job.id, time.time() + self.idem_ttl)
            self.queue.append(job.id)
            self.cond.notify()
            return job, True

    async def get(self, job_id):
        return self.jobs.get(job_id)

    async def take_batch(self, max_size, max_wait_s):
        batch: list[Job] = []
        async with self.cond:
            try:
                await asyncio.wait_for(self.cond.wait_for(lambda: self.queue), 1)
            except asyncio.TimeoutError:
                return batch
            deadline = time.monotonic() + max_wait_s
            while len(batch) < max_size:
                if self.queue:
                    job = self.jobs[self.queue.popleft()]
                    job.status = Status.running
                    job.attempts += 1
                    batch.append(job)
                    continue
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    await asyncio.wait_for(self.cond.wait(), left)
                except asyncio.TimeoutError:
                    break
        return batch

    async def complete(self, job: Job, result: dict[str, Any]):
        job.result = result
        job.status = Status.done
        job.finished_at = time.time()

    async def fail(self, job, error, max_attempts):
        job.error = error
        if job.attempts >= max_attempts:
            job.status = Status.dead
            job.finished_at = time.time()
            self.dead.append(job.id)
            return False
        job.status = Status.queued
        async with self.cond:
            self.queue.append(job.id)
            self.cond.notify()
        return True

    async def depths(self):
        return len(self.queue), len(self.dead)

    async def dead_letters(self, limit=50):
        return [self.jobs[i] for i in self.dead[-limit:]]

    async def close(self):
        pass
