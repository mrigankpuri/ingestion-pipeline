from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.domain.ingestion import DeleteJobResponse, JobStatusResponse
from app.repository.ingestion_jobs import JobRecord
from app.services.ingestion import IngestionService

router = APIRouter(prefix="/api/v1/ingestion", tags=["ingestion"])


def get_ingestion_service(request: Request) -> IngestionService:
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="Ingestion service not initialized")
    return service


def _parse_metadata(metadata_json: str | None) -> dict[str, Any]:
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


def _to_job_status(record: JobRecord) -> JobStatusResponse:
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


@router.post("/uploads", response_model=JobStatusResponse, status_code=202)
async def upload_and_ingest(
    file: UploadFile = File(...),
    namespace: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> JobStatusResponse:
    metadata = _parse_metadata(metadata_json)
    record = await service.submit_upload(upload=file, namespace=namespace, metadata=metadata)
    return _to_job_status(record)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> JobStatusResponse:
    record = service.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _to_job_status(record)


@router.delete("/jobs/{job_id}", response_model=DeleteJobResponse, status_code=202)
async def delete_job(
    job_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> DeleteJobResponse:
    record = await service.request_delete(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return DeleteJobResponse(
        job_id=record.job_id,
        status=record.status,
        completion_event=record.completion_event,
        delete_requested=record.delete_requested,
    )
