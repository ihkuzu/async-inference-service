import asyncio

import httpx
import pytest_asyncio
from httpx import ASGITransport

from inferq.app import create_app
from inferq.config import Settings


@pytest_asyncio.fixture
async def client():
    cfg = Settings(backend="memory", predictor="simulated", embedded_worker=True,
                   max_batch=8, max_wait_ms=5)
    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            yield c


async def _wait(client, job_id):
    for _ in range(200):
        r = await client.get(f"/v1/jobs/{job_id}")
        if r.json()["status"] in ("done", "dead"):
            return r.json()
        await asyncio.sleep(0.01)
    raise AssertionError("job did not finish")


async def test_submit_poll_result(client):
    r = await client.post("/v1/jobs", json={"text": "a short text"})
    assert r.status_code == 202
    assert r.headers["location"] == f"/v1/jobs/{r.json()['id']}"
    job = await _wait(client, r.json()["id"])
    assert job["status"] == "done" and job["result"]["label"] == "short"


async def test_idempotency_key_returns_200_and_same_id(client):
    h = {"Idempotency-Key": "abc"}
    a = await client.post("/v1/jobs", json={"text": "x"}, headers=h)
    b = await client.post("/v1/jobs", json={"text": "x"}, headers=h)
    assert a.status_code == 202 and b.status_code == 200
    assert a.json()["id"] == b.json()["id"]


async def test_validation_and_not_found(client):
    assert (await client.post("/v1/jobs", json={"text": ""})).status_code == 422
    assert (await client.post("/v1/jobs", json={})).status_code == 422
    assert (await client.get("/v1/jobs/nope")).status_code == 404


async def test_failed_job_shows_in_dead_letters(client):
    r = await client.post("/v1/jobs", json={"text": "__fail__"})
    job = await _wait(client, r.json()["id"])
    assert job["status"] == "dead" and job["attempts"] == 3
    dead = (await client.get("/v1/dead-letters")).json()
    assert [d["id"] for d in dead] == [job["id"]]


async def test_health_and_metrics(client):
    r = await client.post("/v1/jobs", json={"text": "hello"})
    await _wait(client, r.json()["id"])
    assert (await client.get("/health")).json()["status"] == "ok"
    text = (await client.get("/metrics")).text
    assert "inferq_jobs_submitted_total 1.0" in text
    assert 'inferq_jobs_finished_total{status="done"} 1.0' in text
    assert "inferq_batch_size_bucket" in text
