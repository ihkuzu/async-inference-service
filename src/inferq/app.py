import asyncio
import contextlib
import logging

from fastapi import FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .backends import make_backend
from .config import Settings
from .metrics import Metrics
from .models import Job, JobRequest
from .predictors import load_predictor
from .worker import Worker

log = logging.getLogger("inferq.api")


def create_app(settings: Settings | None = None, backend=None) -> FastAPI:
    cfg = settings or Settings()
    metrics = Metrics()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = backend or make_backend(
            cfg.backend, cfg.redis_url, idem_ttl_s=cfg.idem_ttl_s,
            job_ttl_s=cfg.job_ttl_s, worker_id=cfg.worker_id,
        )
        task = None
        if cfg.embedded_worker:
            worker = Worker(app.state.backend, load_predictor(cfg.predictor), metrics,
                            cfg.max_batch, cfg.max_wait_ms, cfg.max_attempts)
            task = asyncio.create_task(worker.run())
        yield
        if task:
            worker.stop.set()
            await task
        await app.state.backend.close()

    app = FastAPI(title="inferq", version="0.1.0", lifespan=lifespan)
    app.state.metrics = metrics

    @app.post("/v1/jobs", status_code=202)
    async def submit(req: JobRequest, response: Response,
                     idempotency_key: str | None = Header(default=None, max_length=200)) -> Job:
        job, created = await app.state.backend.submit(req.text, idempotency_key)
        if created:
            metrics.submitted.inc()
        else:
            metrics.deduplicated.inc()
            response.status_code = 200
        response.headers["Location"] = f"/v1/jobs/{job.id}"
        return job

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> Job:
        job = await app.state.backend.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/v1/dead-letters")
    async def dead_letters(limit: int = 50) -> list[Job]:
        return await app.state.backend.dead_letters(min(limit, 500))

    @app.get("/health")
    async def health():
        queued, dead = await app.state.backend.depths()
        return {"status": "ok", "queued": queued, "dead": dead}

    @app.get("/metrics")
    async def prom():
        queued, dead = await app.state.backend.depths()
        metrics.queue_depth.set(queued)
        metrics.dead_depth.set(dead)
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def app_factory() -> FastAPI:
    return create_app()
