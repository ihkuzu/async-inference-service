from .base import Backend
from .memory import MemoryBackend


def make_backend(kind: str, redis_url: str = "", **kw) -> Backend:
    if kind == "memory":
        return MemoryBackend(**kw)
    if kind == "redis":
        from .redis_backend import RedisBackend

        return RedisBackend(redis_url, **kw)
    raise ValueError(f"unknown backend: {kind}")


__all__ = ["Backend", "MemoryBackend", "make_backend"]
