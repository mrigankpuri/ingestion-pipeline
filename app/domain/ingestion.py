from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    DELETE_REQUESTED = "DELETE_REQUESTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DELETED = "DELETED"


ACTIVE_STATUSES = frozenset(
    {JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.DELETE_REQUESTED}
)
TERMINAL_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.DELETED}
)


class JobStatusResponse(BaseModel):
    job_id: str
    file_name: str
    original_file_name: str
    namespace: str | None = None
    status: JobStatus
    completion_event: bool
    delete_requested: bool
    error_message: str | None = None
    result: dict[str, Any] | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


class DeleteJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    completion_event: bool
    delete_requested: bool
