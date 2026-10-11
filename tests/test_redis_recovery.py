from inferq.backends import make_backend
from inferq.models import Status


async def test_crashed_worker_jobs_are_recovered(redis_url):
    first = make_backend("redis", redis_url, worker_id="w1")
    await first.r.flushdb()
    job, _ = await first.submit("lost?", None)
    taken = await first.take_batch(4, 0.01)
    assert [j.id for j in taken] == [job.id]
    assert (await first.depths())[0] == 0
    await first.close()  # worker "crashes" before completing

    second = make_backend("redis", redis_url, worker_id="w1")
    assert await second.recover() == 1
    (again,) = await second.take_batch(4, 0.01)
    assert again.id == job.id and again.attempts == 2 and again.status == Status.running
    await second.close()


async def test_two_workers_never_get_the_same_job(redis_url):
    a = make_backend("redis", redis_url, worker_id="wa")
    b = make_backend("redis", redis_url, worker_id="wb")
    await a.r.flushdb()
    for i in range(40):
        await a.submit(f"t{i}", None)
    import asyncio

    ra, rb = await asyncio.gather(a.take_batch(25, 0.05), b.take_batch(25, 0.05))
    ids = [j.id for j in ra] + [j.id for j in rb]
    assert len(ids) == 40 and len(set(ids)) == 40
    await a.close()
    await b.close()
