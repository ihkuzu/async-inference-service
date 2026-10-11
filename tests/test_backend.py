import asyncio
import time

from inferq.models import Status


async def test_submit_and_get(backend):
    job, created = await backend.submit("hello", None)
    assert created and job.status == Status.queued
    got = await backend.get(job.id)
    assert got.text == "hello"
    assert await backend.get("missing") is None


async def test_idempotency_key_returns_same_job(backend):
    a, created_a = await backend.submit("x", "key-1")
    b, created_b = await backend.submit("x", "key-1")
    assert created_a and not created_b
    assert a.id == b.id
    assert (await backend.depths())[0] == 1


async def test_concurrent_same_key_creates_one_job(backend):
    results = await asyncio.gather(*[backend.submit("x", "same") for _ in range(20)])
    assert len({j.id for j, _ in results}) == 1
    assert sum(created for _, created in results) == 1
    assert (await backend.depths())[0] == 1


async def test_take_batch_respects_max_size_and_order(backend):
    for i in range(5):
        await backend.submit(f"t{i}", None)
    batch = await backend.take_batch(3, 0.05)
    assert [j.text for j in batch] == ["t0", "t1", "t2"]
    assert all(j.status == Status.running and j.attempts == 1 for j in batch)
    rest = await backend.take_batch(10, 0.01)
    assert [j.text for j in rest] == ["t3", "t4"]


async def test_take_batch_waits_for_late_arrivals(backend):
    await backend.submit("first", None)

    async def late():
        await asyncio.sleep(0.03)
        await backend.submit("second", None)

    task = asyncio.create_task(late())
    batch = await backend.take_batch(8, 0.3)
    await task
    assert [j.text for j in batch] == ["first", "second"]


async def test_take_batch_returns_at_deadline(backend):
    await backend.submit("only", None)
    t0 = time.monotonic()
    batch = await backend.take_batch(8, 0.1)
    assert len(batch) == 1
    assert 0.08 < time.monotonic() - t0 < 0.5


async def test_empty_queue_returns_empty_batch(backend):
    assert await backend.take_batch(4, 0.01) == []


async def test_complete(backend):
    await backend.submit("a", None)
    (job,) = await backend.take_batch(1, 0.01)
    await backend.complete(job, {"label": "ok"})
    got = await backend.get(job.id)
    assert got.status == Status.done and got.result == {"label": "ok"}


async def test_fail_requeues_then_dead_letters(backend):
    await backend.submit("a", None)
    (job,) = await backend.take_batch(1, 0.01)
    assert await backend.fail(job, "boom", max_attempts=2) is True
    assert (await backend.depths()) == (1, 0)
    (job,) = await backend.take_batch(1, 0.01)
    assert job.attempts == 2
    assert await backend.fail(job, "boom", max_attempts=2) is False
    assert (await backend.depths()) == (0, 1)
    dead = await backend.dead_letters()
    assert dead[0].status == Status.dead and dead[0].error == "boom"
