from uuid import UUID, uuid4
from datetime import datetime, timezone
from redis.asyncio import Redis
from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from celery.result import AsyncResult

from app.worker.celery_app import celery_app
from app.worker.tasks import detect

from app.config import Settings
from app.schemas import (BatchResult, 
                         JobCreated, 
                         JobStatus, 
                         RejectedFile, 
                         JobInfo, 
                         Prediction)

from app.storage import UploadRejected, save_upload, validate_file

# Celery's task states → API's job status
_STATUS_BY_STATE = {
    "PENDING": JobStatus.QUEUED,      # waiting in the queue (we know it exists: we have its record)
    "STARTED": JobStatus.PROCESSING,  # needs task_track_started=True, which you have
    "RETRY": JobStatus.RETRYING,
    "SUCCESS": JobStatus.SUCCEEDED,
    "FAILURE": JobStatus.FAILED,
}

async def submit_jobs(files: list[UploadFile], settings: Settings, redis: Redis) -> BatchResult:
    """Submit a batch of jobs for processing the uploaded files.

    Each valid file is saved and queued as its own job; invalid files are returned in `rejected`.
    """
    rejected_files = []
    jobs = []
    for file in files:
        try:
            validate_file(file, settings)
        except UploadRejected as e:  # base class: also catches error types added later
            rejected_files.append(RejectedFile(filename=file.filename or "<unknown>", error=e.code, detail=str(e)))
            continue

        job_id = uuid4()
        path = await save_upload(file, job_id, settings)
        await save_job_record(redis, job_id, file, settings)
        await run_in_threadpool(
            detect.apply_async,
            kwargs={"job_id": str(job_id), "path": str(path)},
            task_id=str(job_id),
        )
        jobs.append(JobCreated(job_id=job_id,
                               filename=file.filename,
                               status=JobStatus.QUEUED))

    return BatchResult(accepted=jobs, rejected=rejected_files)

async def save_job_record(redis: Redis, job_id: UUID, file:UploadFile, settings: Settings) -> None:
    """Save the job record in Redis."""
    key = _job_key(job_id)
    await redis.hset(key, mapping={
        "filename": file.filename or "",
        "size_bytes" : file.size or 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    
    await redis.expire(key, settings.result_expires_seconds) # expires with Celery's result

def _job_key(job_id: UUID) -> str:
    return f"job:{job_id}"


async def get_job_record(redis : Redis, job_id: UUID) -> dict | None: #TODO: add a schema for the return type
    record = await redis.hgetall(_job_key(job_id))
    return record or None

async def get_job_info(redis: Redis, job_id: UUID) -> JobInfo | None:
    """Return the job's details and status, or None if no such job exists (or it expired)."""
    record = await get_job_record(redis, job_id)
    if record is None:
        return None

    task = await run_in_threadpool(_read_task, job_id)   # Celery's client is blocking
    status = _STATUS_BY_STATE.get(task.state, JobStatus.QUEUED)

    return JobInfo(
        job_id=job_id,
        filename=record["filename"],
        size_bytes=record["size_bytes"],          # "32044" → 32044, Pydantic converts
        created_at=record["created_at"],
        status=status,
        finished_at=task.date_done,
        error=str(task.result) if status == JobStatus.FAILED else None,
        result=Prediction.model_validate(task.result) if status == JobStatus.SUCCEEDED else None,
    )
    
def _read_task(job_id: UUID) -> AsyncResult:
    task = AsyncResult(str(job_id), app=celery_app)
    task.state                       # read from Redis now (blocking), inside the thread pool
    return task