from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model
    detector: Literal["mock"] = "mock"
    device: Literal["cpu", "cuda"] = "cpu"
    model_weights_path: Path = Path("/models/avdnet.pt")

    # Dynamic batching
    max_batch_size: int = 16
    max_batch_wait_ms: float = 20
    max_queue_size: int = 256

    # Preprocessing
    upload_dir: Path = Path("/data/uploads")
    sample_rate: int = 16_000
    clip_seconds: float = 4.0
    min_tail_seconds: float = 1.0

    # Decision
    clip_threshold: float = 0.5  # a clip is "fake" if its score >= this
    positive_threshold: float = 0.2  # a file is "fake" if >= this share of clips are fake
    max_failed_clip_ratio: float = 0.1  # fail the file if more clips than this fail

    # Mock detector
    mock_batch_overhead: float = 0.5
    mock_per_item_delay: float = 0.05
    mock_failure_rate: float = 0.05

    log_level: str = "INFO"


@lru_cache
def get_config() -> Config:
    return Config()
