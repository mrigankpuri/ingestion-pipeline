from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException, Request

from app.domain.ingestion import DeleteJobResponse, JobStatusResponse
from app.repository.ingestion_jobs import JobRecord
from app.services.ingestion import IngestionService


def get_ingestion_service(request: Request) -> IngestionService:
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="Ingestion service not initialized")
    return service


def parse_metadata_json(metadata_json: str | None) -> dict[str, Any]:
    if not metadata_json:
        return {}
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"metadata_json must be valid JSON: {exc}",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata_json must be a JSON object")
    return parsed


def to_job_status_response(record: JobRecord) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=record.job_id,
        file_name=record.file_name,
        original_file_name=record.original_file_name,
        namespace=record.namespace,
        status=record.status,
        completion_event=record.completion_event,
        delete_requested=record.delete_requested,
        error_message=record.error_message,
        result=record.result,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
    )


def to_delete_response(record: JobRecord) -> DeleteJobResponse:
    return DeleteJobResponse(
        job_id=record.job_id,
        status=record.status,
        completion_event=record.completion_event,
        delete_requested=record.delete_requested,
    )
