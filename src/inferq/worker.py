import asyncio
import logging
import time

from .backends.base import Backend
from .metrics import Metrics
from .models import Job
from .predictors import Predictor

log = logging.getLogger("inferq.worker")


class Worker:
    def __init__(self, backend: Backend, predictor: Predictor, metrics: Metrics,
                 max_batch: int = 16, max_wait_ms: int = 20, max_attempts: int = 3):
        self.backend = backend
        self.predictor = predictor
        self.metrics = metrics
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.max_attempts = max_attempts
        self.stop = asyncio.Event()

    async def _predict(self, texts: list[str]):
        t0 = time.perf_counter()
        try:
            return await asyncio.to_thread(self.predictor.predict_batch, texts)
        finally:
            self.metrics.model_seconds.observe(time.perf_counter() - t0)
            self.metrics.batch_size.observe(len(texts))

    async def _finish(self, job: Job, result: dict):
        await self.backend.complete(job, result)
        self.metrics.finished.labels("done").inc()
        self.metrics.job_seconds.observe(job.finished_at - job.created_at)

    async def _fail(self, job: Job, err: Exception):
        requeued = await self.backend.fail(job, repr(err), self.max_attempts)
        if requeued:
            self.metrics.retried.inc()
        else:
            self.metrics.finished.labels("dead").inc()
            self.metrics.job_seconds.observe(job.finished_at - job.created_at)

    async def process(self, batch: list[Job]):
        try:
            results = await self._predict([j.text for j in batch])
            if len(results) != len(batch):
                raise RuntimeError("predictor returned the wrong number of results")
        except Exception as err:
            if len(batch) == 1:
                await self._fail(batch[0], err)
                return
            # Run the items alone so one bad input does not fail the whole batch.
            log.warning("batch of %d failed (%r), retrying items one by one", len(batch), err)
            for job in batch:
                try:
                    (res,) = await self._predict([job.text])
                    await self._finish(job, res)
                except Exception as item_err:
                    await self._fail(job, item_err)
            return
        for job, res in zip(batch, results, strict=True):
            await self._finish(job, res)

    async def run(self):
        if hasattr(self.backend, "recover"):
            n = await self.backend.recover()
            if n:
                log.info("recovered %d jobs from a previous run", n)
        while not self.stop.is_set():
            batch = await self.backend.take_batch(self.max_batch, self.max_wait)
            if batch:
                await self.process(batch)
            queued, dead = await self.backend.depths()
            self.metrics.queue_depth.set(queued)
            self.metrics.dead_depth.set(dead)
        log.info("worker stopped")
