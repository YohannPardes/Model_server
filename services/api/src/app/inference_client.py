from typing import Any

import httpx

from app.config import get_settings


class InferenceError(Exception):
    """Base: any failure talking to the inference service."""


class InferenceUnavailable(InferenceError):
    """Timeout, connection error or 5xx. Temporary, safe to retry."""


class InferenceRejected(InferenceError):
    """4xx. The request itself is invalid; retrying won't help."""


class InferenceBadResponse(InferenceError):
    """Unparseable response. Likely a bug or version mismatch."""


_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    # Created lazily so each Celery worker process gets its own client after fork
    global _client
    if _client is None:
        settings = get_settings()
        _client = httpx.Client(
            base_url=settings.inference_url,
            timeout=settings.inference_timeout_seconds,
        )
    return _client


def inference_request(method: str, path: str, **kwargs: Any) -> Any:
    """Call the inference service and return the decoded JSON body.

    `kwargs` are passed to `httpx.Client.request` (json=, params=, headers=...).
    Every failure is raised as a subclass of `InferenceError`.
    """
    try:
        response = _get_client().request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as e:
        raise InferenceUnavailable(f"{method} {path} timed out") from e
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        error = InferenceUnavailable if code >= 500 else InferenceRejected
        raise error(f"{method} {path} returned {code}: {e.response.text[:200]}") from e
    except httpx.RequestError as e:
        raise InferenceUnavailable(f"{method} {path} failed: {e!r}") from e
    except ValueError as e:
        raise InferenceBadResponse(f"{method} {path} returned invalid JSON") from e
