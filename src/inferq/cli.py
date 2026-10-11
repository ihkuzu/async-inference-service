import argparse
import asyncio
import logging
import signal

from .config import Settings


async def _run_worker(cfg: Settings):
    from prometheus_client import start_http_server

    from .backends import make_backend
    from .metrics import Metrics
    from .predictors import load_predictor
    from .worker import Worker

    metrics = Metrics()
    start_http_server(cfg.worker_metrics_port, registry=metrics.registry)
    backend = make_backend(cfg.backend, cfg.redis_url, idem_ttl_s=cfg.idem_ttl_s,
                           job_ttl_s=cfg.job_ttl_s, worker_id=cfg.worker_id)
    worker = Worker(backend, load_predictor(cfg.predictor), metrics,
                    cfg.max_batch, cfg.max_wait_ms, cfg.max_attempts)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.stop.set)
    await worker.run()
    await backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="inferq")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("worker", help="run a queue worker")
    api = sub.add_parser("api", help="run the HTTP API")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=int, default=8000)
    b = sub.add_parser("bench", help="batch-size sweep with a simulated model")
    b.add_argument("--backend", default="memory", choices=["memory", "redis"])
    b.add_argument("--redis-url", default="redis://localhost:6379/15")
    b.add_argument("--batches", default="1,4,8,16,32")
    b.add_argument("--max-wait-ms", type=int, default=10)
    b.add_argument("--jobs", type=int, default=600)
    b.add_argument("--rate", type=float, default=0, help="arrivals per second, 0 = all at once")
    b.add_argument("--fixed-ms", type=float, default=20)
    b.add_argument("--per-item-ms", type=float, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = Settings()

    if args.cmd == "worker":
        asyncio.run(_run_worker(cfg))
    elif args.cmd == "api":
        import uvicorn

        uvicorn.run("inferq.app:app_factory", factory=True, host=args.host, port=args.port)
    else:
        from .bench import sweep, table

        batches = [int(x) for x in args.batches.split(",")]
        rows = asyncio.run(sweep(args.backend, args.redis_url, batches, args.max_wait_ms,
                                 args.jobs, args.rate, args.fixed_ms, args.per_item_ms))
        print(table(rows))


if __name__ == "__main__":
    main()
