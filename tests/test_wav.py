from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest
from conftest import fmt_payload, riff_chunk, riff_file

from voxreceipt.constants import MAX_WAV_BYTES, MAX_WAV_FRAMES
from voxreceipt.errors import AudioFormatError, SafetyError
from voxreceipt.wav import inspect_wav, read_pcm16_mono, write_pcm16_mono


def test_round_trip_pcm16(tmp_path: Path, sine: np.ndarray) -> None:
    path = tmp_path / "round.wav"
    write_pcm16_mono(path, sine)
    info = inspect_wav(path, require_official_format=True)
    restored = read_pcm16_mono(path)
    assert info.sample_rate == 48_000
    assert info.channels == 1
    assert info.sample_width_bytes == 2
    assert info.frame_count == sine.size
    assert np.max(np.abs(restored - sine)) <= 1 / 32768


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_writer_rejects_non_finite(tmp_path: Path, bad: float) -> None:
    with pytest.raises(AudioFormatError, match="finite"):
        write_pcm16_mono(tmp_path / "bad.wav", [0.0, bad])


@pytest.mark.parametrize("shape", [np.zeros((2, 2)), np.zeros((1, 2, 3))])
def test_writer_rejects_multidimensional(tmp_path: Path, shape: np.ndarray) -> None:
    with pytest.raises(AudioFormatError, match="one-dimensional"):
        write_pcm16_mono(tmp_path / "bad.wav", shape)


def test_writer_clips_range(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    write_pcm16_mono(path, [-2.0, 2.0])
    values = read_pcm16_mono(path)
    assert values[0] == -1.0
    assert values[1] == 32767 / 32768


def test_writer_refuses_overwrite(tmp_path: Path, sine: np.ndarray) -> None:
    path = tmp_path / "once.wav"
    write_pcm16_mono(path, sine)
    original = path.read_bytes()
    with pytest.raises(SafetyError, match="exists"):
        write_pcm16_mono(path, sine)
    assert path.read_bytes() == original


def test_writer_refuses_symlink(tmp_path: Path, sine: np.ndarray) -> None:
    target = tmp_path / "target"
    target.write_text("keep")
    link = tmp_path / "audio.wav"
    link.symlink_to(target)
    with pytest.raises(SafetyError, match="unsafe"):
        write_pcm16_mono(link, sine)
    assert target.read_text() == "keep"


def test_reader_refuses_symlink(tmp_path: Path, sine: np.ndarray) -> None:
    target = tmp_path / "target.wav"
    write_pcm16_mono(target, sine)
    link = tmp_path / "link.wav"
    link.symlink_to(target)
    with pytest.raises(AudioFormatError, match="unreadable"):
        inspect_wav(link)


def test_data_chunk_before_fmt_is_rejected_for_submission_compatibility(tmp_path: Path) -> None:
    pcm = struct.pack("<hhhh", 0, 100, -100, 32767)
    payload = riff_file([riff_chunk(b"data", pcm), riff_chunk(b"fmt ", fmt_payload())])
    path = tmp_path / "reordered.wav"
    path.write_bytes(payload)
    with pytest.raises(AudioFormatError, match="must follow"):
        inspect_wav(path, require_official_format=True)
    with pytest.raises(AudioFormatError, match="must follow"):
        read_pcm16_mono(path)


def test_unknown_odd_chunk_with_padding_is_supported(tmp_path: Path) -> None:
    payload = riff_file(
        [
            riff_chunk(b"JUNK", b"abc"),
            riff_chunk(b"fmt ", fmt_payload()),
            riff_chunk(b"data", struct.pack("<hh", 1, 2)),
        ]
    )
    path = tmp_path / "odd.wav"
    path.write_bytes(payload)
    assert inspect_wav(path).frame_count == 2


def test_odd_chunk_missing_padding_is_rejected(tmp_path: Path) -> None:
    malformed = riff_chunk(b"JUNK", b"abc", padding=False)
    payload = riff_file(
        [malformed, riff_chunk(b"fmt ", fmt_payload()), riff_chunk(b"data", b"\0\0")]
    )
    path = tmp_path / "no-pad.wav"
    path.write_bytes(payload)
    with pytest.raises(AudioFormatError):
        inspect_wav(path)


@pytest.mark.parametrize("difference", [-1, 1, 1000])
def test_riff_declared_size_mismatch_is_rejected(tmp_path: Path, difference: int) -> None:
    chunks = [riff_chunk(b"fmt ", fmt_payload()), riff_chunk(b"data", b"\0\0")]
    valid = riff_file(chunks)
    declared = len(valid) - 8 + difference
    path = tmp_path / f"size-{difference}.wav"
    path.write_bytes(riff_file(chunks, declared_size=declared))
    with pytest.raises(AudioFormatError, match="RIFF size"):
        inspect_wav(path)


def test_huge_data_declaration_is_rejected_without_allocation(tmp_path: Path) -> None:
    path = tmp_path / "huge.wav"
    huge = riff_chunk(b"data", b"", declared=0xFFFFFFF0)
    path.write_bytes(riff_file([riff_chunk(b"fmt ", fmt_payload()), huge]))
    with pytest.raises(AudioFormatError, match="exceeds"):
        inspect_wav(path)


def test_sparse_file_over_byte_limit_is_rejected_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "sparse.wav"
    path.write_bytes(b"RIFF")
    os.truncate(path, MAX_WAV_BYTES + 1)
    with pytest.raises(AudioFormatError, match="64 MiB"):
        read_pcm16_mono(path)


def test_sparse_file_over_frame_limit_is_rejected_before_sample_read(tmp_path: Path) -> None:
    data_size = (MAX_WAV_FRAMES + 1) * 2
    file_size = 44 + data_size
    header = riff_file(
        [riff_chunk(b"fmt ", fmt_payload()), riff_chunk(b"data", b"", declared=data_size)],
        declared_size=file_size - 8,
    )
    path = tmp_path / "too-many-frames.wav"
    path.write_bytes(header)
    os.truncate(path, file_size)
    with pytest.raises(AudioFormatError, match="frame safety limit"):
        inspect_wav(path)


@pytest.mark.parametrize("cut", [1, 4, 9, 15, 30])
def test_truncated_files_are_rejected(tmp_path: Path, cut: int) -> None:
    valid = riff_file(
        [riff_chunk(b"fmt ", fmt_payload()), riff_chunk(b"data", struct.pack("<hhhh", 1, 2, 3, 4))]
    )
    path = tmp_path / f"truncated-{cut}.wav"
    path.write_bytes(valid[:-cut])
    with pytest.raises(AudioFormatError):
        inspect_wav(path)


def test_trailing_bytes_outside_riff_are_rejected(tmp_path: Path) -> None:
    chunks = [riff_chunk(b"fmt ", fmt_payload()), riff_chunk(b"data", b"\0\0")]
    path = tmp_path / "trailing.wav"
    path.write_bytes(riff_file(chunks, trailing=b"hidden"))
    with pytest.raises(AudioFormatError, match="RIFF size"):
        inspect_wav(path)


@pytest.mark.parametrize(
    ("fmt", "match"),
    [
        (fmt_payload(audio_format=3), "integer PCM"),
        (fmt_payload(block_align=4), "inconsistent"),
        (fmt_payload(byte_rate=1), "inconsistent"),
        (fmt_payload(bits=12), "invalid"),
        (fmt_payload(channels=0), "invalid"),
        (fmt_payload(rate=0), "invalid"),
    ],
)
def test_invalid_fmt_is_rejected(tmp_path: Path, fmt: bytes, match: str) -> None:
    path = tmp_path / "fmt.wav"
    path.write_bytes(riff_file([riff_chunk(b"fmt ", fmt), riff_chunk(b"data", b"")]))
    with pytest.raises(AudioFormatError, match=match):
        inspect_wav(path)


@pytest.mark.parametrize(
    "fmt",
    [fmt_payload(rate=44_100), fmt_payload(channels=2), fmt_payload(bits=8)],
)
def test_non_official_format_is_parseable_but_fails_gate(tmp_path: Path, fmt: bytes) -> None:
    channels, bits = struct.unpack("<H", fmt[2:4])[0], struct.unpack("<H", fmt[14:16])[0]
    align = channels * (bits // 8)
    path = tmp_path / "format.wav"
    path.write_bytes(riff_file([riff_chunk(b"fmt ", fmt), riff_chunk(b"data", b"\0" * align)]))
    assert inspect_wav(path).frame_count == 1
    with pytest.raises(AudioFormatError, match="official format"):
        inspect_wav(path, require_official_format=True)


def test_data_mid_frame_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "partial.wav"
    path.write_bytes(riff_file([riff_chunk(b"fmt ", fmt_payload()), riff_chunk(b"data", b"x")]))
    with pytest.raises(AudioFormatError, match="mid-frame"):
        inspect_wav(path)


@pytest.mark.parametrize("duplicate", [b"fmt ", b"data"])
def test_duplicate_required_chunk_is_rejected(tmp_path: Path, duplicate: bytes) -> None:
    fmt = riff_chunk(b"fmt ", fmt_payload())
    data = riff_chunk(b"data", b"\0\0")
    extra = fmt if duplicate == b"fmt " else data
    path = tmp_path / "duplicate.wav"
    path.write_bytes(riff_file([fmt, data, extra]))
    with pytest.raises(AudioFormatError, match=r"multiple|one valid"):
        inspect_wav(path)


@pytest.mark.parametrize("missing", [b"fmt ", b"data"])
def test_missing_required_chunk_is_rejected(tmp_path: Path, missing: bytes) -> None:
    chunks = []
    if missing != b"fmt ":
        chunks.append(riff_chunk(b"fmt ", fmt_payload()))
    if missing != b"data":
        chunks.append(riff_chunk(b"data", b"\0\0"))
    path = tmp_path / "missing.wav"
    path.write_bytes(riff_file(chunks))
    with pytest.raises(AudioFormatError, match="missing"):
        inspect_wav(path)


def test_non_riff_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "text.wav"
    path.write_text("not audio")
    with pytest.raises(AudioFormatError, match=r"truncated|RIFF"):
        inspect_wav(path)


def test_fifo_or_directory_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "folder.wav"
    directory.mkdir()
    with pytest.raises(AudioFormatError, match="regular"):
        inspect_wav(directory)


def test_zero_sample_wave_is_valid_container(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    write_pcm16_mono(path, [])
    assert inspect_wav(path, require_official_format=True).frame_count == 0
    assert read_pcm16_mono(path).size == 0


def test_writer_mode_is_private(tmp_path: Path, sine: np.ndarray) -> None:
    path = tmp_path / "private.wav"
    write_pcm16_mono(path, sine)
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_wav_hard_link_is_rejected(tmp_path: Path, sine: np.ndarray) -> None:
    original = tmp_path / "original.wav"
    linked = tmp_path / "linked.wav"
    write_pcm16_mono(original, sine)
    os.link(original, linked)
    with pytest.raises(AudioFormatError, match="hard link"):
        inspect_wav(linked)


def test_writer_cleanup_uses_original_directory_fd(
    tmp_path: Path, sine: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    displaced = tmp_path / "displaced"
    destination = output / "result.wav"
    original_fsync = os.fsync
    called = False

    def fail_after_directory_replacement(fd: int) -> None:
        nonlocal called
        if called:
            original_fsync(fd)
            return
        called = True
        output.rename(displaced)
        output.mkdir()
        destination.write_bytes(b"replacement-directory-file")
        raise RuntimeError("simulated durable-write failure")

    monkeypatch.setattr("voxreceipt.wav.os.fsync", fail_after_directory_replacement)
    with pytest.raises(SafetyError, match="written safely"):
        write_pcm16_mono(destination, sine)
    assert destination.read_bytes() == b"replacement-directory-file"
    assert not (displaced / "result.wav").exists()
