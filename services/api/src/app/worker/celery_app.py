from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery("model_server",
                    broker=settings.celery_broker_url,
                    backend=settings.celery_result_backend,
                    include=["app.worker.tasks"])

celery_app.conf.update(
    task_track_started=True,     # report STARTED, not just PENDING → SUCCESS
    task_acks_late=True,         # message is removed only after the task finishes (not lost if worker dies)
    worker_prefetch_multiplier=1,  # each process takes 1 task at a time (fair for slow tasks)
    result_expires=settings.result_expires_seconds,
)
