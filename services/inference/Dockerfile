FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU or CUDA wheel, chosen by docker-compose(.gpu).yml. The CUDA wheels bundle the CUDA
# runtime; the host driver is provided by the NVIDIA Container Toolkit.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# Installed first and on its own layer: it's the largest download and rarely changes
RUN pip install torch --index-url ${TORCH_INDEX_URL}

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 9000

CMD ["uvicorn", "inference_server.main:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
