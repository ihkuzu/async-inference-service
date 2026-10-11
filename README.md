# inferq

A small queue-based inference service. Clients submit a job and get an id back right away; a worker picks jobs off a Redis queue, groups them into batches, runs the model once per batch, and stores the results. It is the pattern behind most "slow model behind an HTTP API" setups, written out in one readable codebase.

```
client ──POST /v1/jobs──▶ API ──▶ Redis queue ──▶ worker (dynamic batching) ──▶ predictor
   ▲                       │                          │
   └──GET /v1/jobs/{id}────┴────── job hash ◀─────────┘      /metrics on API and worker
```

## What it does

- **Async job API.** `POST /v1/jobs` returns `202` with a job id; `GET /v1/jobs/{id}` returns the status and result.
- **Idempotency.** Send an `Idempotency-Key` header and a repeated request returns the original job (`200`) instead of creating a second one. Concurrent requests with the same key create exactly one job.
- **Dynamic batching.** The worker waits for the first job, then keeps collecting until it has `MAX_BATCH` jobs or `MAX_WAIT_MS` has passed, whichever comes first.
- **Retries and a dead-letter queue.** A failing job is requeued up to `MAX_ATTEMPTS` times, then lands in the dead-letter queue (`GET /v1/dead-letters`). If a whole batch fails, the worker re-runs its items one by one, so a single bad input does not take its neighbours down.
- **Crash safety.** Redis jobs are moved into a per-worker processing list and only removed on completion. A restarted worker puts unfinished jobs back on the queue.
- **Graceful shutdown.** On `SIGTERM` the worker finishes the batch it is running and then exits.
- **Metrics.** Prometheus metrics for queue depth, batch size, model time, end-to-end job time, retries and dead letters.
- **Pluggable parts.** Queue backends (`memory`, `redis`) and models (`Predictor` protocol: anything with `predict_batch(list[str]) -> list[dict]`).

## Run it

```bash
docker compose up --build
curl -s -X POST localhost:8000/v1/jobs -H 'content-type: application/json' \
     -H 'Idempotency-Key: demo-1' -d '{"text": "hello world"}'
curl -s localhost:8000/v1/jobs/<id>
```

Without Docker (needs a Redis on localhost, or use `INFERQ_BACKEND=memory INFERQ_EMBEDDED_WORKER=1` for a single process):

```bash
pip install -e ".[dev]"
inferq api &       # http://localhost:8000/docs
inferq worker
```

### Using a real model

The default predictor is a simulation (see the benchmark notes). To serve a scikit-learn pipeline saved with joblib:

```bash
INFERQ_PREDICTOR=sklearn:/path/to/model.joblib inferq worker
```

Any other model works by pointing `INFERQ_PREDICTOR` at `package.module:factory`, where `factory()` returns an object with `predict_batch`.

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `INFERQ_BACKEND` | `memory` | `memory` or `redis` |
| `INFERQ_REDIS_URL` | `redis://localhost:6379/0` | |
| `INFERQ_PREDICTOR` | `simulated` | `simulated`, `sklearn:<path>` or `module:factory` |
| `INFERQ_MAX_BATCH` | `16` | largest batch per model call |
| `INFERQ_MAX_WAIT_MS` | `20` | how long to wait for a batch to fill |
| `INFERQ_MAX_ATTEMPTS` | `3` | tries before a job is dead-lettered |
| `INFERQ_EMBEDDED_WORKER` | `0` | run a worker inside the API process |
| `INFERQ_WORKER_ID` | `worker-1` | names the processing list; keep it stable across restarts and unique per worker |

## Benchmark

`inferq bench` runs the real queue and worker against a simulated model that costs 20 ms per call plus 1 ms per item, which is the shape of a GPU forward pass (large fixed cost, small marginal cost). The numbers below come from one container with a local Redis, 600 jobs, one worker.

**Burst: all 600 jobs arrive at once**

| max_batch | jobs/s | p50 ms | p95 ms |
|---:|---:|---:|---:|
| 1 | 43 | 6955 | 13198 |
| 4 | 133 | 2183 | 4112 |
| 8 | 199 | 1431 | 2689 |
| 16 | 275 | 1002 | 1886 |
| 32 | 347 | 776 | 1404 |

**Steady load: 150 jobs/s arriving**

| max_batch | jobs/s | p50 ms | p95 ms |
|---:|---:|---:|---:|
| 1 | 43 | 3744 | 7183 |
| 4 | 137 | 157 | 270 |
| 8 | 146 | 55 | 74 |
| 16 | 146 | 54 | 73 |

Without batching the worker tops out at about 43 jobs/s (one 21 ms call per job), so under 150 jobs/s the queue grows without bound and latency climbs. With a batch size of 8 it keeps up and p95 stays under 100 ms. Beyond that, a bigger batch cannot help because the arrival rate is already served.

Things to keep in mind when reading this:

- The model is simulated. The container has no GPU and cannot download model weights. The batching curve depends on how much of your real model's cost is fixed per call, so measure your own model before copying settings.
- The benchmark client polls job status every 5 ms, which adds a few milliseconds to every latency and limits the burst numbers. Treat them as relative, not absolute.
- One worker, one machine, no network between the services.

Reproduce with `inferq bench --backend redis --redis-url redis://localhost:6379/15 --jobs 600` (add `--rate 150` for the steady case).

## Design notes

- **Why batch at the worker and not the API?** The API stays stateless and cheap. Batching needs a shared view of the queue, which the worker already has.
- **Redis list + per-worker processing list.** `BLMOVE` moves a job atomically from the queue to the worker's own list, so there is no window where a job exists nowhere. It is the classic reliable-queue pattern. The trade-off is at-least-once delivery: a crash after the model ran but before the result was stored runs the job again. Predictors should be safe to repeat, and for most inference they are.
- **Idempotency via `SET NX`.** The key is claimed before the job is written, so two simultaneous requests cannot both create one. Keys expire after `INFERQ_IDEM_TTL_S`.
- **Errors are isolated per item.** The retry path runs failed batches item by item, so the cost of a poison input is one extra model call per neighbour, not lost work.

## Limitations

- Results and jobs expire after `INFERQ_JOB_TTL_S` (a day by default); there is no long-term store.
- No authentication or rate limiting; put it behind a gateway.
- Scaling to several workers works (each needs its own `INFERQ_WORKER_ID`), but a worker that never comes back leaves its in-flight jobs in its processing list until a worker with the same id starts again.
- The model runs in a thread, so a CPU-bound Python model shares the GIL with the worker loop. Run more worker processes if that matters.

## Tests

```bash
pytest
```

41 tests. Backend behaviour (ordering, batching deadline, idempotency under concurrency, retries, dead letters) runs against both the in-memory backend and a real `redis-server` that the test suite starts itself. Further tests cover the worker (poison-item isolation, transient failures, graceful stop), the HTTP API, metrics, and crash recovery with two competing workers. CI runs lint, the tests on Python 3.10 and 3.12, and a Docker Compose smoke test.

## License

MIT
