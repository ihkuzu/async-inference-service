from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.submitted = Counter("inferq_jobs_submitted_total", "Jobs accepted", registry=r)
        self.deduplicated = Counter("inferq_jobs_deduplicated_total", "Repeated idempotency keys", registry=r)
        self.finished = Counter("inferq_jobs_finished_total", "Jobs finished", ["status"], registry=r)
        self.retried = Counter("inferq_jobs_retried_total", "Jobs requeued after a failure", registry=r)
        self.batch_size = Histogram(
            "inferq_batch_size", "Items per model call", buckets=(1, 2, 4, 8, 16, 32, 64, 128), registry=r
        )
        self.model_seconds = Histogram("inferq_model_seconds", "Model call duration", registry=r)
        self.job_seconds = Histogram(
            "inferq_job_seconds", "Time from submit to finish", registry=r,
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
        )
        self.queue_depth = Gauge("inferq_queue_depth", "Jobs waiting", registry=r)
        self.dead_depth = Gauge("inferq_dead_letter_depth", "Jobs in the dead-letter queue", registry=r)
