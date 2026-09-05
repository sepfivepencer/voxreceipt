from __future__ import annotations

import struct
from collections.abc import Sequence

import numpy as np
import pytest


@pytest.fixture
def sine() -> np.ndarray:
    time = np.arange(4_800, dtype=np.float64) / 48_000
    return 0.3 * np.sin(2 * np.pi * 440 * time)


def fmt_payload(
    *,
    audio_format: int = 1,
    channels: int = 1,
    rate: int = 48_000,
    bits: int = 16,
    block_align: int | None = None,
    byte_rate: int | None = None,
) -> bytes:
    align = channels * (bits // 8) if block_align is None else block_align
    bytes_per_second = rate * align if byte_rate is None else byte_rate
    return struct.pack("<HHIIHH", audio_format, channels, rate, bytes_per_second, align, bits)


def riff_chunk(
    chunk_id: bytes,
    payload: bytes,
    *,
    declared: int | None = None,
    padding: bool = True,
) -> bytes:
    size = len(payload) if declared is None else declared
    pad = b"\0" if padding and len(payload) % 2 else b""
    return struct.pack("<4sI", chunk_id, size) + payload + pad


def riff_file(
    chunks: Sequence[bytes], *, declared_size: int | None = None, trailing: bytes = b""
) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    size = len(body) if declared_size is None else declared_size
    return b"RIFF" + struct.pack("<I", size) + body + trailing
