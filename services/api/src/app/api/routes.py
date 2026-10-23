from uuid import UUID

from fastapi import APIRouter, Depends, Response, UploadFile, status

from app.config import Settings, get_settings

router = APIRouter(prefix="/api/v1")


@router.post("/jobs", status_code=202)
async def create_job(file: UploadFile): ...


@router.post("/jobs/batch", status_code=202)
async def create_batch(files: list[UploadFile]): ...


@router.get("/jobs/{job_id}")
async def get_job(job_id: UUID): ...


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: UUID): ...


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: UUID): ...


@router.get("/health")
async def health(): ...
