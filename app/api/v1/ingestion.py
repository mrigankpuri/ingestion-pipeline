from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.v1.ingestion_helpers import (
    get_ingestion_service,
    parse_metadata_json,
    to_delete_response,
    to_job_status_response,
)
from app.domain.ingestion import DeleteJobResponse, JobStatusResponse
from app.services.ingestion import IngestionService

router = APIRouter(prefix="/api/v1/ingestion", tags=["ingestion"])


@router.post("/uploads", response_model=JobStatusResponse, status_code=202)
async def upload_and_ingest(
    file: UploadFile = File(...),
    namespace: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> JobStatusResponse:
    metadata = parse_metadata_json(metadata_json)
    record = await service.submit_upload(upload=file, namespace=namespace, metadata=metadata)
    return to_job_status_response(record)


@router.get("/jobs", response_model=list[JobStatusResponse])
def get_all_jobs(
    service: IngestionService = Depends(get_ingestion_service),
) -> list[JobStatusResponse]:
    records = service.get_all_jobs()
    return [_to_job_status(record) for record in records]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> JobStatusResponse:
    record = service.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return to_job_status_response(record)


@router.delete("/jobs/{job_id}", response_model=DeleteJobResponse, status_code=202)
async def delete_job(
    job_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> DeleteJobResponse:
    record = await service.request_delete(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return to_delete_response(record)
