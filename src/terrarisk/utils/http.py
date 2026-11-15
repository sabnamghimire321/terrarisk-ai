"""
Shared HTTP helper for all data-ingestion modules. Centralizes
retry/backoff/timeout behavior so every ingestion module gets the
same resilient behavior instead of each reimplementing it slightly
differently.
"""

import time
import requests

from src.terrarisk.config import REQUEST_MAX_RETRIES, REQUEST_TIMEOUT_SECONDS

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

def _request_with_retry(
    url: str,
    params: dict | None = None,
    max_retries: int | None = None,
) -> requests.Response:
    """
    GET with retry and exponential backoff, but only on genuinely
    transient failures: network errors, or the retryable status codes
    above. A non-retryable status (e.g. 404) fails immediately via
    raise_for_status() rather than being retried like a transient one.

    `max_retries` lets a specific caller ask for more patience (e.g.
    open_meteo_flood.py, whose requests are heavy enough to be more
    likely to hit a shared rate-limit window) without changing the
    default behavior for cheap, fast calls elsewhere.
    """
    retries = max_retries if max_retries is not None else REQUEST_MAX_RETRIES
    last_exception: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except requests.RequestException as exc:
            last_exception = exc
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
            continue

        if response.status_code in RETRYABLE_STATUS_CODES:
            last_exception = requests.HTTPError(
                f"Retryable status {response.status_code} from {url}"
            )
            if attempt < retries:
                backoff_seconds = min(2 ** (attempt - 1), 60)
                print(
                    f"  Got {response.status_code}, retrying in "
                    f"{backoff_seconds}s (attempt {attempt}/{retries})..."
                )
                time.sleep(backoff_seconds)
            continue

        response.raise_for_status()
        return response

    raise requests.RequestException(
        f"Failed to GET {url} after {retries} attempts"
    ) from last_exception

def get_json(
    url: str, params: dict | None = None, max_retries: int | None = None
) -> dict:
    return _request_with_retry(url, params, max_retries).json()

def get_text(
    url: str, params: dict | None = None, max_retries: int | None = None
) -> str:
    return _request_with_retry(url, params, max_retries).text