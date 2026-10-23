from app.inference_client import InferenceUnavailable, inference_request
from app.worker.celery_app import celery_app


@celery_app.task(
    bind=True,                              # gives access to `self` (task id, retries count)
    autoretry_for=(InferenceUnavailable,),  # retry only temporary errors
    retry_backoff=True,                     # 1s, 2s, 4s, 8s…
    max_retries=5,
)

def detect(self, job_id: str, path: str) -> dict:
    return inference_request("POST", "/predict", json={"path": path, "job_id": job_id})
