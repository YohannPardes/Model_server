from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, status
from redis.asyncio import Redis
from app.storage import FileTooLarge, UnsupportedFileType
from app.config import Settings, get_settings
from app.jobs import submit_jobs, get_job_info
from app.schemas import BatchResult, JobCreated, JobInfo, JobStatus, Prediction

router = APIRouter(prefix="/api/v1")

# HTTP status for a rejected single-file upload; any other error code → 400
_STATUS_BY_ERROR = {
    UnsupportedFileType.code: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    FileTooLarge.code: 413,  # Starlette renamed this constant across versions; the number is stable
}

def get_redis(request: Request) -> Redis:
    return request.app.state.redis

@router.post("/jobs", status_code=202, response_model=JobCreated)
async def create_job(file: UploadFile, settings: Settings = Depends(get_settings), redis: Redis = Depends(get_redis)):
    """Create a new job for processing the uploaded file."""
    # processing the job
    batch_result = await submit_jobs([file], settings, redis)  # Submit the job to the worker

    if batch_result.rejected:
        rejected = batch_result.rejected[0]
        raise HTTPException(_STATUS_BY_ERROR.get(rejected.error, status.HTTP_400_BAD_REQUEST), rejected.detail)

    return batch_result.accepted[0]

@router.post("/jobs/batch", status_code=202, response_model=BatchResult)
async def create_batch(files: list[UploadFile], response: Response, settings: Settings = Depends(get_settings), redis: Redis = Depends(get_redis)):
    """Queue every valid file as its own job and report the invalid ones."""
    batch_result = await submit_jobs(files, settings, redis)  # Submit each file as a separate job

    if not batch_result.accepted:  # nothing was queued: the request as a whole failed
        response.status_code = status.HTTP_400_BAD_REQUEST

    return batch_result

@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: UUID, redis: Redis = Depends(get_redis)):
    info = await get_job_info(redis, job_id)
    if info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")
    return info


@router.get("/jobs/{job_id}/result", response_model=Prediction)
async def get_job_result(job_id: UUID, redis: Redis = Depends(get_redis)):
    info = await get_job_info(redis, job_id)
    if info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")
    if info.status != JobStatus.SUCCEEDED:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job is {info.status}, no result available")
    return info.result

@router.get("/health")
async def health(): ...
