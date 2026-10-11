import asyncio

import pytest

from inferq.metrics import Metrics
from inferq.models import Status
from inferq.worker import Worker


class CountingPredictor:
    def __init__(self):
        self.calls: list[int] = []

    def predict_batch(self, texts):
        self.calls.append(len(texts))
        if any("__fail__" in t for t in texts):
            raise ValueError("bad input")
        return [{"echo": t} for t in texts]


class FlakyPredictor:
    def __init__(self, fail_times):
        self.left = fail_times

    def predict_batch(self, texts):
        if self.left > 0:
            self.left -= 1
            raise RuntimeError("temporary")
        return [{"ok": True} for _ in texts]


async def _wait_all(backend, ids, timeout=5.0):
    async def poll():
        while True:
            jobs = [await backend.get(i) for i in ids]
            if all(j.status in (Status.done, Status.dead) for j in jobs):
                return jobs
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(poll(), timeout)


@pytest.fixture
def metrics():
    return Metrics()


async def _run(worker, coro):
    task = asyncio.create_task(worker.run())
    try:
        return await coro
    finally:
        worker.stop.set()
        await asyncio.wait_for(task, 5)


async def test_jobs_are_batched(backend, metrics):
    pred = CountingPredictor()
    ids = [(await backend.submit(f"t{i}", None))[0].id for i in range(20)]
    worker = Worker(backend, pred, metrics, max_batch=8, max_wait_ms=20)
    jobs = await _run(worker, _wait_all(backend, ids))
    assert [j.result["echo"] for j in jobs] == [f"t{i}" for i in range(20)]
    assert max(pred.calls) <= 8
    assert len(pred.calls) < 20


async def test_poison_item_does_not_fail_neighbours(backend, metrics):
    pred = CountingPredictor()
    texts = ["a", "b", "__fail__", "c"]
    ids = [(await backend.submit(t, None))[0].id for t in texts]
    worker = Worker(backend, pred, metrics, max_batch=8, max_wait_ms=20, max_attempts=2)
    jobs = await _run(worker, _wait_all(backend, ids))
    status = {j.text: j.status for j in jobs}
    assert status == {"a": Status.done, "b": Status.done, "c": Status.done, "__fail__": Status.dead}
    assert (await backend.depths())[1] == 1


async def test_transient_failure_is_retried(backend, metrics):
    ids = [(await backend.submit("x", None))[0].id]
    worker = Worker(backend, FlakyPredictor(2), metrics, max_batch=1, max_wait_ms=1, max_attempts=3)
    (job,) = await _run(worker, _wait_all(backend, ids))
    assert job.status == Status.done and job.attempts == 3
    assert metrics.retried._value.get() == 2


async def test_gives_up_after_max_attempts(backend, metrics):
    ids = [(await backend.submit("x", None))[0].id]
    worker = Worker(backend, FlakyPredictor(99), metrics, max_batch=1, max_wait_ms=1, max_attempts=2)
    (job,) = await _run(worker, _wait_all(backend, ids))
    assert job.status == Status.dead and job.attempts == 2


async def test_wrong_result_count_is_an_error(backend, metrics):
    class Short:
        def predict_batch(self, texts):
            return [{}]

    ids = [(await backend.submit(t, None))[0].id for t in ("a", "b")]
    worker = Worker(backend, Short(), metrics, max_batch=2, max_wait_ms=20, max_attempts=1)
    jobs = await _run(worker, _wait_all(backend, ids))
    # Alone, each item gets one result, so both succeed after the split.
    assert all(j.status == Status.done for j in jobs)


async def test_graceful_stop_finishes_running_batch(backend, metrics):
    class Slow:
        def predict_batch(self, texts):
            import time

            time.sleep(0.15)
            return [{"ok": 1} for _ in texts]

    ids = [(await backend.submit(f"t{i}", None))[0].id for i in range(3)]
    worker = Worker(backend, Slow(), metrics, max_batch=3, max_wait_ms=10)
    task = asyncio.create_task(worker.run())
    await asyncio.sleep(0.07)
    worker.stop.set()
    await asyncio.wait_for(task, 5)
    jobs = [await backend.get(i) for i in ids]
    assert all(j.status == Status.done for j in jobs)
