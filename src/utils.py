from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utcnow().isoformat()


def estimate_tokens(text: str) -> int:
    words = text.split()
    if not words:
        return 1
    return max(1, int(len(words) * 1.3))


def chunked(iterable: Sequence[T] | Iterable[T], size: int) -> Iterator[list[T]]:
    if size <= 0:
        raise ValueError("size must be positive")
    chunk: list[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class RateLimiter:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = asyncio.Lock()
        self._last_call: float = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            delay = self.interval - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_call = time.monotonic()


def setup_logging(log_file: Path, verbose: bool = False) -> None:
    ensure_directory(log_file.parent)
    handlers = [logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


__all__ = [
    "RateLimiter",
    "chunked",
    "ensure_directory",
    "estimate_tokens",
    "iso_now",
    "setup_logging",
    "utcnow",
]
