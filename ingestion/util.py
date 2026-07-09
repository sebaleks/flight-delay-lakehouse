"""Small shared helpers for ingestion jobs."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def download_with_retries(
    url: str,
    dest: Path,
    attempts: int = 5,
    connect_timeout: int = 20,
    read_timeout: int = 300,
) -> None:
    """Stream `url` to `dest`, retrying transient network/server errors.

    Writes to a .part file and renames on completion so an interrupted
    download never leaves a plausible-looking partial file at `dest`.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(connect_timeout, read_timeout)) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return
        except (requests.RequestException, OSError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status is None or status in RETRYABLE_STATUS
            if attempt == attempts or not retryable:
                raise
            delay = min(60, 2**attempt) + random.uniform(0, 1)
            log.warning(
                "download attempt %d/%d for %s failed (%s); retrying in %.0fs",
                attempt,
                attempts,
                url,
                exc,
                delay,
            )
            time.sleep(delay)
