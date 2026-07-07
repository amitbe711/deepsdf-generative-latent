"""Lightweight status logging for long grid / training runs."""

from __future__ import annotations

import time
from datetime import datetime


def status(message: str, *, prefix: str | None = None) -> None:
    """Print a timestamped line and flush immediately (Colab-friendly)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = f"[{prefix}] " if prefix else ""
    print(f"[{ts}] {tag}{message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


class Phase:
    """Log start/end of a pipeline phase with elapsed time."""

    def __init__(self, name: str, *, prefix: str | None = None) -> None:
        self.name = name
        self.prefix = prefix
        self._t0 = 0.0

    def __enter__(self) -> Phase:
        self._t0 = time.time()
        status(f"start {self.name}", prefix=self.prefix)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.time() - self._t0
        if exc_type is None:
            status(f"done {self.name} ({format_duration(elapsed)})", prefix=self.prefix)
        else:
            status(
                f"failed {self.name} after {format_duration(elapsed)}: {exc_type.__name__}",
                prefix=self.prefix,
            )
        return False
