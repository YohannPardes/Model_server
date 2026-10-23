from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["fake", "real"]


class PredictRequest(BaseModel):
    path: str = Field(description="Audio file on the shared uploads volume")


class ModelPrediction(BaseModel):
    score: float = Field(ge=0, le=1, description="Model probability that the clip is fake")

class ClipPrediction(BaseModel):
    index: int = Field(description="Clip position in the file (0 = first 4 s)")
    score: float = Field(ge=0, le=1, description="Model probability that the clip is fake")
    label: Label


class Prediction(BaseModel):
    label: Label = Field(description="Final vote over all clips")
    fake_ratio: float = Field(ge=0, le=1, description="Share of clips labelled fake")
    failed_clip_count: int = Field(description="Number of clips that failed to be processed")
    num_clips: int
    clips: list[ClipPrediction]
    model: str = Field(description="Detector that produced the prediction, e.g. 'mock-v1'")
    inference_ms: float = Field(description="Preprocessing + model time, in milliseconds")


class HealthResponse(BaseModel):
    status: str
    detector: str
    device: str
