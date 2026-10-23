import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from inference_server.batcher import Batcher
from inference_server.config import get_config
from inference_server.loader import load_model
from inference_server.pipeline import Pipeline, TooManyFailedClips
from inference_server.preprocessing import AudioError
from inference_server.schemas import HealthResponse, Prediction, PredictRequest

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # before: load the detector once (loader.py), start the batcher loop (batcher.py)
    config = get_config()
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logger.info(
        "Starting inference server: detector=%s device=%s max_batch_size=%d "
        "max_batch_wait_ms=%s max_queue_size=%d",
        config.detector,
        config.device,
        config.max_batch_size,
        config.max_batch_wait_ms,
        config.max_queue_size,
    )

    start = time.perf_counter()
    model = load_model()  # Load the model once
    logger.info("Model loaded in %.0f ms", (time.perf_counter() - start) * 1000)

    batcher = Batcher(model, config)
    app.state.Pipeline = Pipeline(batcher, config)
    await batcher.start()  # Start the batcher loop
    try:
        yield
    finally:
        # after: stop the batcher loop, even if the app exits with an error
        logger.info("Shutting down inference server")
        await batcher.stop()


app = FastAPI(
    title="Inference Server",
    description="Internal model server: one model, dynamic batching. Called by the workers only.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/predict")
async def predict(request: PredictRequest) -> Prediction:
    # Map expected errors to the status codes the worker's retry logic relies on.
    # Anything else is a bug: FastAPI returns 500 and the worker retries.
    filename = Path(request.path).name
    try:
        return await app.state.Pipeline.process_request(request)
    except AudioError as e:
        logger.warning("Rejected %s (422): %s", filename, e)
        raise HTTPException(422, str(e)) from e  # bad file → worker: no retry
    except asyncio.QueueFull as e:
        logger.warning("Rejected %s (503): inference queue full", filename)
        raise HTTPException(503, "Inference queue full") from e  # busy → worker: retry
    except TooManyFailedClips as e:
        logger.warning("Failed %s (503): %s", filename, e)
        raise HTTPException(503, str(e)) from e  # model trouble → worker: retry


@app.get("/health")
async def health() -> HealthResponse:
    batcher = app.state.Pipeline.batcher
    if batcher._task is None or batcher._task.done():
        raise HTTPException(503, "Batcher not running")
    config = get_config()
    return HealthResponse(status="ok", detector=config.detector, device=config.device)

