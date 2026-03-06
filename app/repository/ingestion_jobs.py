from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from app.domain.ingestion import JobStatus


@dataclass(slots=True)
class JobRecord:
    job_id: str
    file_name: str
    original_file_name: str
    content_type: str | None
    namespace: str | None
    metadata: dict[str, Any]
    stored_file_path: str
    artifact_path: str | None
    status: JobStatus
    completion_event: bool
    delete_requested: bool
    error_message: str | None
    result: dict[str, Any] | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None


class IngestionRepository:
    def __init__(self, sqlite_path: Path) -> None:
        self._lock = RLock()
        self._connection = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    original_file_name TEXT NOT NULL,
                    content_type TEXT,
                    namespace TEXT,
                    metadata_json TEXT NOT NULL,
                    stored_file_path TEXT NOT NULL,
                    artifact_path TEXT,
                    status TEXT NOT NULL,
                    completion_event INTEGER NOT NULL DEFAULT 0,
                    delete_requested INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
                """
            )
            self._connection.commit()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _decode_json(raw_value: str | None, default: Any) -> Any:
        if raw_value in (None, ""):
            return default
        return json.loads(raw_value)

    def _row_to_record(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            file_name=row["file_name"],
            original_file_name=row["original_file_name"],
            content_type=row["content_type"],
            namespace=row["namespace"],
            metadata=self._decode_json(row["metadata_json"], {}),
            stored_file_path=row["stored_file_path"],
            artifact_path=row["artifact_path"],
            status=JobStatus(row["status"]),
            completion_event=bool(row["completion_event"]),
            delete_requested=bool(row["delete_requested"]),
            error_message=row["error_message"],
            result=self._decode_json(row["result_json"], None),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def create_job(
        self,
        *,
        job_id: str,
        file_name: str,
        original_file_name: str,
        content_type: str | None,
        namespace: str | None,
        metadata: dict[str, Any],
        stored_file_path: str,
    ) -> JobRecord:
        now = self._now_iso()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id,
                    file_name,
                    original_file_name,
                    content_type,
                    namespace,
                    metadata_json,
                    stored_file_path,
                    status,
                    completion_event,
                    delete_requested,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    file_name,
                    original_file_name,
                    content_type,
                    namespace,
                    json.dumps(metadata),
                    stored_file_path,
                    JobStatus.QUEUED.value,
                    0,
                    0,
                    now,
                    now,
                ),
            )
            self._connection.commit()
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def mark_processing(self, job_id: str) -> JobRecord | None:
        now = self._now_iso()
        with self._lock:
            self._connection.execute(
                """
                UPDATE ingestion_jobs
                SET
                    status = ?,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE job_id = ? AND status = ? AND delete_requested = 0
                """,
                (
                    JobStatus.PROCESSING.value,
                    now,
                    now,
                    job_id,
                    JobStatus.QUEUED.value,
                ),
            )
            self._connection.commit()
        return self.get_job(job_id)

    def mark_delete_requested(self, job_id: str) -> JobRecord | None:
        now = self._now_iso()
        with self._lock:
            self._connection.execute(
                """
                UPDATE ingestion_jobs
                SET
                    delete_requested = 1,
                    status = CASE
                        WHEN status IN (?, ?) THEN ?
                        ELSE status
                    END,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    JobStatus.QUEUED.value,
                    JobStatus.PROCESSING.value,
                    JobStatus.DELETE_REQUESTED.value,
                    now,
                    job_id,
                ),
            )
            self._connection.commit()
        return self.get_job(job_id)

    def mark_completed(
        self,
        job_id: str,
        *,
        result: dict[str, Any],
        artifact_path: str,
    ) -> bool:
        now = self._now_iso()
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE ingestion_jobs
                SET
                    status = ?,
                    completion_event = 1,
                    delete_requested = 0,
                    result_json = ?,
                    artifact_path = ?,
                    error_message = NULL,
                    completed_at = ?,
                    updated_at = ?
                WHERE job_id = ? AND delete_requested = 0
                """,
                (
                    JobStatus.COMPLETED.value,
                    json.dumps(result),
                    artifact_path,
                    now,
                    now,
                    job_id,
                ),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def mark_failed(self, job_id: str, error_message: str) -> JobRecord | None:
        now = self._now_iso()
        with self._lock:
            self._connection.execute(
                """
                UPDATE ingestion_jobs
                SET
                    status = ?,
                    completion_event = 1,
                    error_message = ?,
                    completed_at = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    JobStatus.FAILED.value,
                    error_message,
                    now,
                    now,
                    job_id,
                ),
            )
            self._connection.commit()
        return self.get_job(job_id)

    def mark_deleted(self, job_id: str) -> JobRecord | None:
        now = self._now_iso()
        with self._lock:
            self._connection.execute(
                """
                UPDATE ingestion_jobs
                SET
                    status = ?,
                    completion_event = 1,
                    delete_requested = 0,
                    result_json = NULL,
                    artifact_path = NULL,
                    error_message = NULL,
                    completed_at = COALESCE(completed_at, ?),
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    JobStatus.DELETED.value,
                    now,
                    now,
                    job_id,
                ),
            )
            self._connection.commit()
        return self.get_job(job_id)

    def should_delete(self, job_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT delete_requested FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return bool(row and row["delete_requested"])

    def close(self) -> None:
        with self._lock:
            self._connection.close()
