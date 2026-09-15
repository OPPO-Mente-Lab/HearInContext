# Copyright (c) 2026 OPPO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable manifest handling shared by training and decoding entrypoints."""
import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor" / "Qwen3-ASR"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def resolve_audio(row, data_root):
    relative = Path(row["audio"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected audio path relative to data root: {relative}")
    path = data_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path.resolve())


@contextmanager
def writer_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
