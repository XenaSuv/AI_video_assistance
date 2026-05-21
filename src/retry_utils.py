"""Shared retry helpers for external HTTP and OpenAI API calls."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI

import requests
from loguru import logger
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


def _is_retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        code = getattr(exc.response, "status_code", None)
        return code in (429, 500, 502, 503, 504)
    return False


def _log_http_retry(retry_state: RetryCallState) -> None:
    logger.warning(
        f"HTTP retry {retry_state.attempt_number}: {retry_state.outcome.exception()}"  # type: ignore[union-attr]
    )


@retry(  # type: ignore[misc]
    retry=retry_if_exception(_is_retryable_http),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    before_sleep=_log_http_retry,
    reraise=True,
)
def http_get(url: str, **kwargs: Any) -> requests.Response:
    """GET with automatic retry on transient errors."""
    resp = requests.get(url, **kwargs)
    resp.raise_for_status()
    return resp


@retry(  # type: ignore[misc]
    retry=retry_if_exception(_is_retryable_http),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    before_sleep=_log_http_retry,
    reraise=True,
)
def http_post(url: str, **kwargs: Any) -> requests.Response:
    """POST with automatic retry on transient errors."""
    resp = requests.post(url, **kwargs)
    resp.raise_for_status()
    return resp


def make_openai_client(**kwargs: Any) -> "OpenAI":
    """Return an OpenAI client with increased retry count."""
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key, max_retries=3, **kwargs)
