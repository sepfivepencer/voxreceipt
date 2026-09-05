"""Strict, bounded RIFF/WAVE parsing and exclusive PCM16 writing."""

from __future__ import annotations

import os
import stat
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

from voxreceipt.constants import (
    CHANNELS,
    MAX_WAV_BYTES,
    MAX_WAV_FRAMES,
    PCM_COMPRESSION,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
)
from voxreceipt.errors import AudioFormatError, SafetyError
from voxreceipt.models import WavInfo
from voxreceipt.safeio import CreatedFile, DirectoryHandle, remove_created_files

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class _ParsedWave:
    info: WavInfo
    data_offset: int
    data_size: int


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise AudioFormatError("WAV file is truncated")
    return value


def _parse_wave(handle: BinaryIO, file_size: int) -> _ParsedWave:
    if file_size > MAX_WAV_BYTES:
        raise AudioFormatError("WAV file exceeds the 64 MiB safety limit")
    if file_size < 12:
        raise AudioFormatError("WAV file is truncated")
    riff, declared_size, wave_tag = struct.unpack("<4sI4s", _read_exact(handle, 12))
    if riff != b"RIFF" or wave_tag != b"WAVE":
        raise AudioFormatError("file is not a RIFF/WAVE stream")
    riff_end = declared_size + 8
    if riff_end != file_size:
        raise AudioFormatError("RIFF size does not match the file length")

    fmt: tuple[int, int, int, int, int, int] | None = None
    data_offset: int | None = None
    data_size: int | None = None
    data_before_fmt = False
    while handle.tell() < riff_end:
        if riff_end - handle.tell() < 8:
            raise AudioFormatError("WAV chunk header is truncated")
        chunk_id, chunk_size = struct.unpack("<4sI", _read_exact(handle, 8))
        payload_start = handle.tell()
        payload_end = payload_start + chunk_size
        padded_end = payload_end + (chunk_size & 1)
        if payload_end < payload_start or padded_end > riff_end:
            raise AudioFormatError("WAV chunk length exceeds the RIFF container")
        if chunk_id == b"fmt ":
            if fmt is not None or chunk_size < 16:
                raise AudioFormatError("WAV must contain one valid fmt chunk")
            fmt = struct.unpack("<HHIIHH", _read_exact(handle, 16))
        elif chunk_id == b"data":
            if data_offset is not None:
                raise AudioFormatError("WAV contains multiple data chunks")
            if fmt is None:
                data_before_fmt = True
            data_offset = payload_start
            data_size = chunk_size
        handle.seek(payload_end)
        if chunk_size & 1:
            _read_exact(handle, 1)

    if handle.tell() != riff_end or fmt is None or data_offset is None or data_size is None:
        raise AudioFormatError("WAV is missing required fmt or data chunks")
    if data_before_fmt:
        raise AudioFormatError("WAV data chunk must follow the fmt chunk")
    audio_format, channels, rate, byte_rate, block_align, bits = fmt
    if audio_format != 1:
        raise AudioFormatError("WAV must use integer PCM encoding")
    if channels <= 0 or rate <= 0 or bits <= 0 or bits % 8:
        raise AudioFormatError("WAV fmt values are invalid")
    expected_align = channels * (bits // 8)
    if block_align != expected_align or byte_rate != rate * block_align:
        raise AudioFormatError("WAV fmt values are inconsistent")
    if data_size % block_align:
        raise AudioFormatError("WAV data chunk ends mid-frame")
    info = WavInfo(
        sample_rate=rate,
        channels=channels,
        sample_width_bytes=bits // 8,
        frame_count=data_size // block_align,
        compression=PCM_COMPRESSION,
    )
    if info.frame_count > MAX_WAV_FRAMES:
        raise AudioFormatError("WAV file exceeds the 12000000-frame safety limit")
    return _ParsedWave(info=info, data_offset=data_offset, data_size=data_size)


def _open_regular(path: Path) -> tuple[BinaryIO, int]:
    if path.name in {"", ".", ".."}:
        raise AudioFormatError("WAV filename is invalid")
    try:
        with DirectoryHandle.open(path.parent) as directory:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path.name, flags, dir_fd=directory.fd)
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                os.close(fd)
                raise AudioFormatError("audio input must be a regular file")
            if metadata.st_nlink != 1:
                os.close(fd)
                raise AudioFormatError("audio input must have exactly one hard link")
            if metadata.st_size > MAX_WAV_BYTES:
                os.close(fd)
                raise AudioFormatError("WAV file exceeds the 64 MiB safety limit")
            return os.fdopen(fd, "rb", closefd=True), metadata.st_size
    except AudioFormatError:
        raise
    except (OSError, SafetyError) as exc:
        raise AudioFormatError("WAV file is unreadable") from exc


def _official_problems(info: WavInfo) -> list[str]:
    problems: list[str] = []
    if info.sample_rate != SAMPLE_RATE:
        problems.append("sample rate")
    if info.channels != CHANNELS:
        problems.append("channel count")
    if info.sample_width_bytes != SAMPLE_WIDTH_BYTES:
        problems.append("sample width")
    if info.compression != PCM_COMPRESSION:
        problems.append("compression")
    return problems


def inspect_wav(path: Path, *, require_official_format: bool = False) -> WavInfo:
    """Read a bounded WAV header and optionally enforce 48 kHz PCM16 mono."""

    handle, size = _open_regular(path)
    try:
        with handle:
            parsed = _parse_wave(handle, size)
    except AudioFormatError:
        raise
    except (OSError, EOFError, struct.error) as exc:
        raise AudioFormatError("WAV file is unreadable") from exc
    if require_official_format and (problems := _official_problems(parsed.info)):
        raise AudioFormatError("WAV violates official format: " + ", ".join(problems))
    return parsed.info


def read_pcm16_mono(path: Path, *, sample_rate: int = SAMPLE_RATE) -> FloatArray:
    """Read strict PCM16 mono samples as finite float64 values in [-1, 1]."""

    handle, size = _open_regular(path)
    try:
        with handle:
            parsed = _parse_wave(handle, size)
            info = parsed.info
            if (
                info.sample_rate != sample_rate
                or info.channels != CHANNELS
                or info.sample_width_bytes != SAMPLE_WIDTH_BYTES
            ):
                raise AudioFormatError("WAV is not the requested PCM16 mono format")
            handle.seek(parsed.data_offset)
            raw = _read_exact(handle, parsed.data_size)
    except AudioFormatError:
        raise
    except (OSError, EOFError, struct.error) as exc:
        raise AudioFormatError("WAV samples are unreadable") from exc
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def _pcm16_bytes(samples: npt.ArrayLike, sample_rate: int) -> bytes:
    if sample_rate <= 0:
        raise AudioFormatError("sample rate must be positive")
    try:
        values = np.asarray(samples, dtype=np.float64)
    except Exception:
        raise AudioFormatError("audio samples could not be converted safely") from None
    if values.ndim != 1:
        raise AudioFormatError("audio samples must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise AudioFormatError("audio samples must be finite")
    if values.size > MAX_WAV_FRAMES or 44 + values.size * SAMPLE_WIDTH_BYTES > MAX_WAV_BYTES:
        raise AudioFormatError("audio output exceeds the documented WAV safety limits")
    clipped = np.clip(values, -1.0, 32767.0 / 32768.0)
    return np.rint(clipped * 32768.0).astype("<i2").tobytes()


def write_pcm16_mono_at(
    directory: DirectoryHandle,
    name: str,
    samples: npt.ArrayLike,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> CreatedFile:
    """Create one direct-child PCM16 WAV through a held directory descriptor."""

    if name in {"", ".", ".."} or "\0" in name or "/" in name or os.sep in name:
        raise SafetyError("output filename is invalid")
    pcm = _pcm16_bytes(samples, sample_rate)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    created: CreatedFile | None = None
    try:
        fd = os.open(name, flags, 0o600, dir_fd=directory.fd)
        metadata = os.fstat(fd)
        created = CreatedFile(name, metadata.st_dev, metadata.st_ino)
        with os.fdopen(fd, "wb", closefd=True) as raw_handle:
            fd = None
            with wave.open(raw_handle, "wb") as handle:
                handle.setnchannels(CHANNELS)
                handle.setsampwidth(SAMPLE_WIDTH_BYTES)
                handle.setframerate(sample_rate)
                handle.writeframes(pcm)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
            current = os.stat(name, dir_fd=directory.fd, follow_symlinks=False)
            opened = os.fstat(raw_handle.fileno())
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or opened.st_nlink != 1
                or (current.st_dev, current.st_ino) != (created.device, created.inode)
                or (opened.st_dev, opened.st_ino) != (created.device, created.inode)
            ):
                raise OSError("audio output changed during write")
        return created
    except FileExistsError as exc:
        raise SafetyError("audio output exists or parent is unsafe") from exc
    except Exception as exc:
        if created is not None:
            remove_created_files([created], directory)
        raise SafetyError("audio output could not be written safely") from exc
    finally:
        if fd is not None:
            os.close(fd)


def write_pcm16_mono(path: Path, samples: npt.ArrayLike, *, sample_rate: int = SAMPLE_RATE) -> None:
    """Create a PCM16 mono WAV without following links or overwriting a file."""

    try:
        with DirectoryHandle.open(path.parent) as directory:
            write_pcm16_mono_at(directory, path.name, samples, sample_rate=sample_rate)
    except SafetyError:
        raise
