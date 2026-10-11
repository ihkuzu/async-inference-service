import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Status(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    dead = "dead"


class Job(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: Status = Status.queued
    text: str
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    created_at: float = Field(default_factory=time.time)
    finished_at: float | None = None


class JobRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
