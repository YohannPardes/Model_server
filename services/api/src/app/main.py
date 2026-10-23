from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from app.api.routes import router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)   # make sure the volume path exists
    app.state.redis = Redis.from_url(settings.redis_url)      # one shared client
    yield                                                     # ← the app runs here
    await app.state.redis.aclose()                            # clean shutdown


app = FastAPI(
    title="Model Server",
    description="Asynchronous audio analysis…",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router) # include the API routes