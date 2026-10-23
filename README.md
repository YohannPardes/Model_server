# Model Server

FastAPI service for asynchronous audio analysis. Uploads are queued with Celery/Redis and processed by worker processes, which all send requests to a single inference server that owns the model (mocked by default) and batches requests dynamically. Tasks are monitored with Flower. A minimal web UI and a Python SDK are included.

<!-- TODO: demo GIF (docs/) -->

## Quick start

**Requirements:** Docker with Docker Compose v2.24+.

```bash
cp .env.example .env          # optional, defaults work without it
docker compose up --build
```

Everything runs on **CPU** by default, so this works on any machine. No GPU needed.

| Service  | URL                          | Purpose                          |
|----------|------------------------------|----------------------------------|
| Web UI   | http://localhost:8080        | Upload audio, see results        |
| API docs | http://localhost:8000/docs   | Interactive OpenAPI (Swagger UI) |
| Flower   | http://localhost:5555        | Celery task / worker monitoring  |

Stop with `Ctrl+C`, then `docker compose down` (add `-v` to also delete uploaded files).

## Running on GPU (optional)

GPU support is an **opt-in override**: only the `inference` service needs a GPU, and keeping it out of the base file means the default setup never fails on machines without one.

**Requirements:**
- NVIDIA GPU with a recent driver
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (Linux), or Docker Desktop with the WSL2 backend (Windows, built in)

Check that Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Then start the stack with both compose files:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

What the override changes, for the **`inference` service only**:

| | CPU (default) | GPU override |
|---|---|---|
| PyTorch wheel | `cpu` | `cu124` (CUDA runtime bundled in the wheel) |
| Image tag | `model-server-inference:cpu` | `model-server-inference:gpu` |
| `DEVICE` env | `cpu` | `cuda` |
| GPU reservation | none | 1 × NVIDIA GPU |

The API, Celery workers and Flower use a separate lightweight image without PyTorch. They never touch the model, so they're unaffected by the GPU override.

> If your driver needs a different CUDA build, change `TORCH_INDEX_URL` in `docker-compose.gpu.yml` (e.g. `cu121`, `cu126`). See [pytorch.org](https://pytorch.org/get-started/locally/).

### Troubleshooting

- **`could not select device driver "nvidia"`**: the NVIDIA Container Toolkit isn't installed, or Docker wasn't restarted after installing it. Run the `nvidia-smi` check above.
- **Inference logs show `DEVICE=cuda` but `torch.cuda.is_available()` is `False`**: the CPU image was used. Rebuild with `--build` and make sure both `-f` files are passed.

## Architecture

```
                         ┌──────────────┐
 Browser / SDK ─────────►│  api :8000   │── saves file ──► [uploads volume]
   ▲      (poll status)  └──────┬───────┘                        │
   │                            │ enqueue                         │ read-only
   │                     ┌──────▼───────┐                         │
   └─────────────────────│    redis     │◄──── flower :5555       │
         task result     └──────┬───────┘                         │
                                │                                 │
                ┌───────────────┼───────────────┐                 │
          ┌─────▼────┐    ┌─────▼────┐    ┌─────▼────┐            │
          │ worker 1 │    │ worker 2 │ …  │ worker N │  (Celery, CPU)
          └─────┬────┘    └─────┬────┘    └─────┬────┘            │
                └───────── HTTP /predict ───────┘                 │
                                │                                 │
                      ┌─────────▼──────────┐                      │
                      │ inference :9000    │◄─────────────────────┘
                      │ 1 process, 1 model │
                      │ dynamic batching   │  ── GPU (optional)
                      └────────────────────┘
```

1. `api` stores the upload on a shared volume and enqueues a Celery task, returning a `task_id`.
2. One of N `worker` processes picks it up and calls `inference` over HTTP (internal network only).
3. `inference` queues incoming requests from all workers and runs them through the model in batches (up to `MAX_BATCH_SIZE`, waiting at most `MAX_BATCH_WAIT_MS`).
4. The result goes back to the worker, which stores it in Redis; the client polls `GET /detect/{task_id}`.

<!-- TODO: short description of each service -->

## API

<!-- TODO: endpoint summary -->

## Python SDK

<!-- TODO: usage example -->

## Configuration

<!-- TODO: table of .env variables -->

## Running tests

<!-- TODO -->

## Design decisions & trade-offs

- **A single inference server, separate from the Celery workers**: if each Celery process loaded the model, N processes would mean N copies in GPU memory and no batching across requests. Instead, one process owns the model and batches requests from all workers. Celery still handles what it's good at (queueing, retries, task state, monitoring), and workers scale independently of the GPU.
- **Dynamic batching** (`MAX_BATCH_SIZE` / `MAX_BATCH_WAIT_MS`): it trades a few milliseconds of latency for much higher GPU throughput under load. The mock models this as `overhead + per_item × batch_size`, so the benefit is visible.
- **Two images**: only `inference` installs PyTorch; the api, worker and flower image stays small and builds fast.
- **GPU as an opt-in compose override**: the base `docker-compose.yml` must run anywhere (reviewers, CI, laptops). GPU-specific settings live in `docker-compose.gpu.yml` and only touch `inference`.
- **CUDA via PyTorch wheels, not an `nvidia/cuda` base image**: the `cu*` wheels ship the CUDA runtime; the host driver is injected by the container toolkit. This keeps one inference Dockerfile for both modes, switched by a build arg.
- **Custom server rather than Triton / TorchServe / Ray Serve**: those are what I'd use in production, but a small hand-written batcher keeps the take-home self-contained and shows the mechanism.

<!-- TODO: Celery + Redis, shared upload volume, mock model, model/ import path -->

## Future work

<!-- TODO -->
