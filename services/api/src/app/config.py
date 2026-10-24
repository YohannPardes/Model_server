from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Celery / Redis
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    result_expires_seconds: int = 3600
    # Job records (filename, size, created_at). Own db number so they don't mix with Celery's data
    redis_url: str = "redis://redis:6379/2"

    # Uploads
    upload_dir: Path = Path("/data/uploads")
    max_upload_mb: int = 20
    allowed_extensions: str = "wav,mp3,flac,ogg"

    # Inference service
    inference_url: str = "http://inference:9000"
    inference_timeout_seconds: float = 60

    log_level: str = "INFO"
    

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",")}


@lru_cache
def get_settings() -> Settings:
    return Settings()
