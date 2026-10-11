import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(f"INFERQ_{name}", default)


@dataclass
class Settings:
    backend: str = field(default_factory=lambda: _env("BACKEND", "memory"))
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    predictor: str = field(default_factory=lambda: _env("PREDICTOR", "simulated"))
    max_batch: int = field(default_factory=lambda: int(_env("MAX_BATCH", "16")))
    max_wait_ms: int = field(default_factory=lambda: int(_env("MAX_WAIT_MS", "20")))
    max_attempts: int = field(default_factory=lambda: int(_env("MAX_ATTEMPTS", "3")))
    idem_ttl_s: int = field(default_factory=lambda: int(_env("IDEM_TTL_S", "86400")))
    job_ttl_s: int = field(default_factory=lambda: int(_env("JOB_TTL_S", "86400")))
    embedded_worker: bool = field(default_factory=lambda: _env("EMBEDDED_WORKER", "0") == "1")
    worker_id: str = field(default_factory=lambda: _env("WORKER_ID", "worker-1"))
    worker_metrics_port: int = field(default_factory=lambda: int(_env("WORKER_METRICS_PORT", "9100")))
