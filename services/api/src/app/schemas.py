from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

Label = Literal["fake", "real"]


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# Mirrors the inference server's response. Duplicated on purpose: the services share
# an HTTP contract, not code.
class ClipPrediction(BaseModel):
    index: int = Field(description="Clip position in the file (0 = first 4 s)")
    score: float = Field(ge=0, le=1, description="Model probability that the clip is fake")
    label: Label


class Prediction(BaseModel):
    label: Label = Field(description="Final vote over all clips")
    fake_ratio: float = Field(ge=0, le=1, description="Share of clips labelled fake")
    num_clips: int
    clips: list[ClipPrediction]
    model: str
    inference_ms: float


class JobCreated(BaseModel):
    job_id: UUID
    filename: str
    status: JobStatus


class JobInfo(BaseModel):
    job_id: UUID
    batch_id: UUID | None = None
    filename: str
    size_bytes: int
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = Field(default=None, description="Set when status is 'failed'")
    result: Prediction | None = Field(default=None, description="Set when status is 'succeeded'")


class BatchCreated(BaseModel):
    batch_id: UUID
    jobs: list[JobCreated]


class BatchInfo(BaseModel):
    batch_id: UUID
    created_at: datetime
    total: int
    counts: dict[JobStatus, int] = Field(description="Number of jobs per status")
    jobs: list[JobInfo]


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    redis: bool
