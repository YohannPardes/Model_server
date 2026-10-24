FROM python:3.12-slim

WORKDIR /app

COPY ./src ./src
COPY pyproject.toml .

RUN pip install .

CMD ["uvicorn", "--host", "0.0.0.0", "app.main:app", "--port", "8000"]