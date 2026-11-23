"""Filesystem primitives for write-once pipeline checkpoints."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from uuid import uuid4


@contextmanager
def lock_run(directory: Path):
    """Hold a nonblocking directory lock for the lifetime of an execution."""
    import fcntl

    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process is already writing to this run: {directory}") from exc
        yield fd
    finally:
        # An active child inherits the descriptor and keeps the lock if its
        # parent exits before the child. Do not explicitly unlock it here.
        os.close(fd)


def publish_json(path: Path, payload: object) -> None:
    """Publish complete JSON atomically and fail if the destination exists."""
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    stream = temporary.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
