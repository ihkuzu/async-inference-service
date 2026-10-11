import asyncio
import time
from typing import Any

import redis.asyncio as aioredis

from ..models import Job, Status

# Each worker moves jobs into its own processing list, so a crash does not lose them.
Q = "inferq:queue"
DEAD = "inferq:dead"


def _jkey(job_id: str) -> str:
    return f"inferq:job:{job_id}"


class RedisBackend:
    def __init__(self, url: str, idem_ttl_s: int = 86400, job_ttl_s: int = 86400,
                 worker_id: str = "worker-1", **_):
        self.r = aioredis.from_url(url, decode_responses=True)
        self.idem_ttl = idem_ttl_s
        self.job_ttl = job_ttl_s
        self.proc = f"inferq:processing:{worker_id}"

    async def _save(self, job: Job):
        await self.r.set(_jkey(job.id), job.model_dump_json(), ex=self.job_ttl)

    async def _load(self, job_id: str) -> Job | None:
        raw = await self.r.get(_jkey(job_id))
        return Job.model_validate_json(raw) if raw else None

    async def submit(self, text, idem_key):
        job = Job(text=text)
        if idem_key:
            # Claim the key first; whoever wins creates the job.
            won = await self.r.set(f"inferq:idem:{idem_key}", job.id, nx=True, ex=self.idem_ttl)
            if not won:
                existing = await self.r.get(f"inferq:idem:{idem_key}")
                found = await self._load(existing) if existing else None
                if found:
                    return found, False
                await self.r.set(f"inferq:idem:{idem_key}", job.id, ex=self.idem_ttl)
        await self._save(job)
        await self.r.lpush(Q, job.id)
        return job, True

    async def get(self, job_id):
        return await self._load(job_id)

    async def recover(self) -> int:
        """Put jobs left in this worker's processing list by a crash back on the queue."""
        n = 0
        while await self.r.lmove(self.proc, Q, "RIGHT", "RIGHT"):
            n += 1
        return n

    async def _pop(self, timeout: float) -> str | None:
        if timeout <= 0:
            return await self.r.lmove(Q, self.proc, "RIGHT", "LEFT")
        return await self.r.blmove(Q, self.proc, timeout, "RIGHT", "LEFT")

    async def take_batch(self, max_size, max_wait_s):
        batch: list[Job] = []
        first = await self._pop(1)
        if first is None:
            return batch
        ids = [first]
        deadline = time.monotonic() + max_wait_s
        while len(ids) < max_size:
            left = deadline - time.monotonic()
            nxt = await self._pop(0)
            if nxt is not None:
                ids.append(nxt)
                continue
            if left <= 0:
                break
            await asyncio.sleep(min(0.002, left))
        for jid in ids:
            job = await self._load(jid)
            if job is None:
                await self.r.lrem(self.proc, 1, jid)
                continue
            job.status = Status.running
            job.attempts += 1
            await self._save(job)
            batch.append(job)
        return batch

    async def complete(self, job: Job, result: dict[str, Any]):
        job.result = result
        job.status = Status.done
        job.finished_at = time.time()
        await self._save(job)
        await self.r.lrem(self.proc, 1, job.id)

    async def fail(self, job, error, max_attempts):
        job.error = error
        requeue = job.attempts < max_attempts
        if requeue:
            job.status = Status.queued
        else:
            job.status = Status.dead
            job.finished_at = time.time()
        await self._save(job)
        async with self.r.pipeline(transaction=True) as p:
            p.lrem(self.proc, 1, job.id)
            p.lpush(Q if requeue else DEAD, job.id)
            await p.execute()
        return requeue

    async def depths(self):
        return await self.r.llen(Q), await self.r.llen(DEAD)

    async def dead_letters(self, limit=50):
        ids = await self.r.lrange(DEAD, 0, limit - 1)
        jobs = [await self._load(i) for i in ids]
        return [j for j in jobs if j]

    async def close(self):
        await self.r.aclose()
