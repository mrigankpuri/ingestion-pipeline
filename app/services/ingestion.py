from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import IngestionSettings
from app.domain.ingestion import ACTIVE_STATUSES, JobStatus
from app.repository.ingestion_jobs import IngestionRepository, JobRecord
from app.services.processors.langchain_processor import IngestionProcessor
from app.services.storage import LocalFileStore
from app.services.vector_indexer import VectorIndexer


class DeleteRequestedError(Exception):
    pass


class IngestionService:
    def __init__(
        self,
        *,
        settings: IngestionSettings,
        repository: IngestionRepository,
        file_store: LocalFileStore,
        vector_indexer: VectorIndexer,
        processor: IngestionProcessor,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.file_store = file_store
        self.vector_indexer = vector_indexer
        self.processor = processor
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._tasks_lock = asyncio.Lock()

    async def submit_upload(
        self,
        *,
        upload: UploadFile,
        namespace: str | None,
        metadata: dict[str, object] | None,
    ) -> JobRecord:
        original_file_name = Path(upload.filename or "uploaded_file.bin").name
        if not original_file_name:
            original_file_name = "uploaded_file.bin"

        job_id = uuid4().hex
        stored_name = f"{job_id}_{original_file_name}"
        stored_path = await self.file_store.persist_upload(upload=upload, stored_name=stored_name)

        record = self.repository.create_job(
            job_id=job_id,
            file_name=stored_name,
            original_file_name=original_file_name,
            content_type=upload.content_type,
            namespace=namespace,
            metadata=dict(metadata or {}),
            stored_file_path=str(stored_path),
        )
        await self._start_job(job_id)
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.repository.get_job(job_id)

    def get_all_jobs(self) -> list[JobRecord]:
        return self.repository.get_all_jobs()

    async def request_delete(self, job_id: str) -> JobRecord | None:
        record = self.repository.get_job(job_id)
        if record is None:
            return None
        if record.status == JobStatus.DELETED:
            return record

        current = self.repository.mark_delete_requested(job_id)
        if current is None:
            return None

        task = await self._get_task(job_id)
        if current.status in ACTIVE_STATUSES:
            if task is None or task.done():
                await self._cleanup_and_mark_deleted(job_id)
            return self.repository.get_job(job_id)

        await self._cleanup_and_mark_deleted(job_id)
        return self.repository.get_job(job_id)

    async def shutdown(self) -> None:
        async with self._tasks_lock:
            running_tasks = [task for task in self._tasks.values() if not task.done()]
        for task in running_tasks:
            task.cancel()
        if running_tasks:
            await asyncio.gather(*running_tasks, return_exceptions=True)

    async def _get_task(self, job_id: str) -> asyncio.Task[None] | None:
        async with self._tasks_lock:
            return self._tasks.get(job_id)

    async def _start_job(self, job_id: str) -> None:
        async with self._tasks_lock:
            existing = self._tasks.get(job_id)
            if existing is not None and not existing.done():
                return
            task = asyncio.create_task(self._run_job(job_id))
            self._tasks[job_id] = task
            task.add_done_callback(lambda _: self._tasks.pop(job_id, None))

    async def _run_job(self, job_id: str) -> None:
        record = self.repository.get_job(job_id)
        if record is None:
            return
        if record.delete_requested:
            await self._cleanup_and_mark_deleted(job_id)
            return

        processing_record = self.repository.mark_processing(job_id)
        if processing_record is None:
            return
        if processing_record.delete_requested or processing_record.status == JobStatus.DELETE_REQUESTED:
            await self._cleanup_and_mark_deleted(job_id)
            return

        extra_paths: list[str | Path | None] = []
        try:
            await self._delay_with_delete_checks(job_id)

            current = self.repository.get_job(job_id)
            if current is None:
                return

            outcome = await asyncio.to_thread(
                self.processor.process,
                job_id=job_id,
                file_path=Path(current.stored_file_path),
                namespace=current.namespace,
                metadata=current.metadata,
                content_type=current.content_type,
            )
            if self.repository.should_delete(job_id):
                raise DeleteRequestedError()

            await self._delay_with_delete_checks(job_id)

            indexing_result = await asyncio.to_thread(
                self.vector_indexer.upsert_documents,
                job_id=job_id,
                namespace=current.namespace,
                documents=outcome.documents,
            )
            vector_store_result = {
                "provider": indexing_result.provider,
                "index_name": indexing_result.index_name,
                "namespace": indexing_result.namespace,
                "indexed_vectors": indexing_result.indexed_vectors,
            }
            outcome.result["vector_store"] = vector_store_result
            outcome.artifact_payload["vector_store"] = vector_store_result

            if self.repository.should_delete(job_id):
                raise DeleteRequestedError()

            artifact_path = await asyncio.to_thread(
                self.file_store.write_artifact,
                job_id,
                outcome.artifact_payload,
            )
            extra_paths.append(artifact_path)

            completed = self.repository.mark_completed(
                job_id=job_id,
                result=outcome.result,
                artifact_path=str(artifact_path),
            )
            if not completed:
                await self._cleanup_and_mark_deleted(job_id, extra_paths=extra_paths)
        except DeleteRequestedError:
            await self._cleanup_and_mark_deleted(job_id, extra_paths=extra_paths)
        except asyncio.CancelledError:
            if self.repository.should_delete(job_id):
                await self._cleanup_and_mark_deleted(job_id, extra_paths=extra_paths)
            else:
                self.file_store.cleanup_paths(extra_paths)
                self.repository.mark_failed(job_id, "Job interrupted during service shutdown")
            raise
        except Exception as exc:
            if self.repository.should_delete(job_id):
                await self._cleanup_and_mark_deleted(job_id, extra_paths=extra_paths)
            else:
                self.file_store.cleanup_paths(extra_paths)
                self.repository.mark_failed(job_id, str(exc))

    async def _delay_with_delete_checks(self, job_id: str) -> None:
        remaining = self.settings.processing_delay_seconds
        while remaining > 0:
            if self.repository.should_delete(job_id):
                raise DeleteRequestedError()
            tick = min(self.settings.delete_poll_interval_seconds, remaining)
            await asyncio.sleep(tick)
            remaining -= tick

    async def _cleanup_and_mark_deleted(
        self,
        job_id: str,
        *,
        extra_paths: Sequence[str | Path | None] | None = None,
    ) -> None:
        record = self.repository.get_job(job_id)
        cleanup_targets: list[str | Path | None] = list(extra_paths or [])
        if record is not None:
            with suppress(Exception):
                await asyncio.to_thread(
                    self.vector_indexer.delete_job,
                    job_id=job_id,
                )
            cleanup_targets.extend([record.stored_file_path, record.artifact_path])

        await asyncio.to_thread(self.file_store.cleanup_paths, cleanup_targets)
        if record is not None:
            self.repository.mark_deleted(job_id)
