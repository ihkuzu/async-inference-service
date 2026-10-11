import asyncio
import statistics
import time

from .backends import make_backend
from .metrics import Metrics
from .predictors import SimulatedPredictor
from .worker import Worker


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


async def run_once(backend_kind: str, redis_url: str, max_batch: int, max_wait_ms: int,
                   n: int, rate: float, fixed_ms: float, per_item_ms: float) -> dict:
    backend = make_backend(backend_kind, redis_url, worker_id=f"bench-{max_batch}")
    if backend_kind == "redis":
        await backend.r.flushdb()
    worker = Worker(backend, SimulatedPredictor(fixed_ms, per_item_ms), Metrics(),
                    max_batch, max_wait_ms)
    task = asyncio.create_task(worker.run())
    ids = []
    t0 = time.perf_counter()
    for i in range(n):
        job, _ = await backend.submit(f"request number {i}", None)
        ids.append(job.id)
        if rate > 0:
            await asyncio.sleep(max(0.0, t0 + (i + 1) / rate - time.perf_counter()))
    pending = set(ids)
    done = {}
    while pending:
        for jid in list(pending):
            j = await backend.get(jid)
            if j and j.finished_at:
                done[jid] = j
                pending.discard(jid)
        await asyncio.sleep(0.005)
    wall = time.perf_counter() - t0
    worker.stop.set()
    await task
    await backend.close()
    lat = [(j.finished_at - j.created_at) * 1000 for j in done.values()]
    return {
        "max_batch": max_batch, "jobs": n, "throughput": n / wall,
        "p50": statistics.median(lat), "p95": pct(lat, 0.95), "p99": pct(lat, 0.99),
    }


async def sweep(backend_kind: str, redis_url: str, batches: list[int], max_wait_ms: int,
                n: int, rate: float, fixed_ms: float, per_item_ms: float) -> list[dict]:
    rows = []
    for b in batches:
        rows.append(await run_once(backend_kind, redis_url, b, max_wait_ms, n, rate, fixed_ms, per_item_ms))
    return rows


def table(rows: list[dict]) -> str:
    head = "| max_batch | jobs/s | p50 ms | p95 ms | p99 ms |\n|---:|---:|---:|---:|---:|"
    body = "\n".join(
        f"| {r['max_batch']} | {r['throughput']:.0f} | {r['p50']:.0f} | {r['p95']:.0f} | {r['p99']:.0f} |"
        for r in rows
    )
    return head + "\n" + body
