from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from voxreceipt.adapters import AudioRestorer, IdentityAdapter
from voxreceipt.cli import _make_demo, _run, build_parser, main
from voxreceipt.errors import AdapterError, AudioFormatError, MetricError, SafetyError
from voxreceipt.preflight import evaluate_directories, preflight_directories, restore_directory
from voxreceipt.privacy import Pseudonymizer
from voxreceipt.signals import degrade_signal, synthetic_speechlike
from voxreceipt.wav import write_pcm16_mono


def roots(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    return source, output


@pytest.mark.parametrize(
    ("source_frames", "output_frames", "valid"),
    [
        (5_000, 8_000, True),
        (5_000, 2_000, True),
        (5_000, 8_001, False),
        (5_000, 1_999, False),
        (5_000, 5_000, True),
    ],
)
def test_sample_delta_positive_and_negative_boundaries(
    tmp_path: Path, source_frames: int, output_frames: int, valid: bool
) -> None:
    source, output = roots(tmp_path)
    write_pcm16_mono(source / "private-name.wav", np.zeros(source_frames))
    write_pcm16_mono(output / "private-name.wav", np.zeros(output_frames))
    report = preflight_directories(
        source,
        output,
        pseudonyms=Pseudonymizer(b"k" * 32, persistent=True),
    )
    assert report["summary"]["valid"] is valid
    delta = output_frames - source_frames
    assert report["files"][0]["sample_delta"] == delta
    assert report["files"][0]["absolute_sample_delta"] == abs(delta)


def test_preflight_finds_missing_and_extra_without_names(tmp_path: Path) -> None:
    source, output = roots(tmp_path)
    write_pcm16_mono(source / "secret-input.wav", [0.0])
    write_pcm16_mono(output / "secret-extra.wav", [0.0])
    report_path = tmp_path / "report.json"
    report = preflight_directories(source, output, report_path=report_path)
    serialized = report_path.read_text()
    assert report["summary"]["valid"] is False
    assert report["summary"]["failure_count"] == 2
    assert "secret-input" not in serialized
    assert "secret-extra" not in serialized
    assert str(tmp_path) not in serialized


def test_preflight_rejects_invalid_output_container_in_report(tmp_path: Path) -> None:
    source, output = roots(tmp_path)
    write_pcm16_mono(source / "x.wav", [0.0])
    (output / "x.wav").write_bytes(b"broken")
    report = preflight_directories(source, output)
    assert report["files"][0]["errors"] == ["output_format_or_container_invalid"]


def test_preflight_empty_input_is_invalid(tmp_path: Path) -> None:
    source, output = roots(tmp_path)
    report = preflight_directories(source, output)
    assert report["summary"]["valid"] is False
    assert report["summary"]["checked_count"] == 0


def test_report_cannot_live_in_audio_directory(tmp_path: Path) -> None:
    source, output = roots(tmp_path)
    with pytest.raises(SafetyError, match="inside"):
        preflight_directories(source, output, report_path=output / "report.json")


def test_restore_identity_and_receipt_privacy(tmp_path: Path, sine: np.ndarray) -> None:
    source = tmp_path / "input"
    source.mkdir()
    write_pcm16_mono(source / "alice-sensitive.wav", sine)
    output = tmp_path / "output"
    report_path = tmp_path / "restore.json"
    times = iter([10.0, 10.01])
    report = restore_directory(
        source,
        output,
        IdentityAdapter(),
        report_path=report_path,
        pseudonyms=Pseudonymizer(b"x" * 32, persistent=True),
        clock=lambda: next(times),
    )
    assert report["summary"] == {"valid": True, "processed_count": 1}
    assert report["files"][0]["latency"]["elapsed_ms"] == pytest.approx(10.0)
    serialized = report_path.read_text()
    assert "alice-sensitive" not in serialized
    assert str(tmp_path) not in serialized
    assert "0.017263" not in serialized
    assert (output / "alice-sensitive.wav").exists()


class CountingAdapter:
    name = "counting-test"
    version = "1"
    kind = "test-only"

    def __init__(self) -> None:
        self.calls = 0

    def restore(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        return samples


def test_restore_reserves_report_before_adapter_processing(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    write_pcm16_mono(source / "x.wav", [0.0, 0.1])
    report = tmp_path / "existing.json"
    report.write_text("keep")
    adapter = CountingAdapter()
    output = tmp_path / "output"
    with pytest.raises(SafetyError, match="already exists"):
        restore_directory(source, output, adapter, report_path=report)
    assert adapter.calls == 0
    assert list(output.iterdir()) == []
    assert report.read_text() == "keep"


class FailsOnSecondAdapter:
    name = "failure-test"
    version = "1"
    kind = "test-only"

    def __init__(self, sensitive_path: Path) -> None:
        self.calls = 0
        self.sensitive_path = sensitive_path

    def restore(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError(f"private input at {self.sensitive_path}")
        return samples


def test_restore_sanitizes_adapter_exception_and_rolls_back(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    write_pcm16_mono(source / "a.wav", [0.0, 0.1])
    write_pcm16_mono(source / "b.wav", [0.0, 0.1])
    output = tmp_path / "output"
    report = tmp_path / "report.json"
    with pytest.raises(AdapterError) as raised:
        restore_directory(
            source,
            output,
            FailsOnSecondAdapter(tmp_path / "private" / "speaker.wav"),
            report_path=report,
        )
    assert str(raised.value) == "adapter execution or output conversion failed"
    assert str(tmp_path) not in str(raised.value)
    assert raised.value.__suppress_context__ is True
    assert list(output.iterdir()) == []
    assert not report.exists()


class ArrayConversionFailure:
    def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
        raise RuntimeError("/private/speaker/array-source")


class BadArrayAdapter:
    name = "array-failure-test"
    version = "1"
    kind = "test-only"

    def restore(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return ArrayConversionFailure()  # type: ignore[return-value]


def test_restore_sanitizes_array_conversion_exception(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    write_pcm16_mono(source / "x.wav", [0.0, 0.1])
    output = tmp_path / "output"
    with pytest.raises(AdapterError) as raised:
        restore_directory(source, output, BadArrayAdapter())
    assert str(raised.value) == "adapter execution or output conversion failed"
    assert "private" not in str(raised.value)
    assert list(output.iterdir()) == []


def test_cli_adapter_failure_has_no_traceback_or_sensitive_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    write_pcm16_mono(source / "x.wav", [0.0, 0.1])
    output = tmp_path / "output"
    report = tmp_path / "report.json"
    adapter = FailsOnSecondAdapter(tmp_path / "private" / "speaker.wav")
    adapter.calls = 1

    def load_failure(spec: str, *, trust_custom_code: bool) -> FailsOnSecondAdapter:
        assert spec == "private-adapter"
        assert trust_custom_code is True
        return adapter

    monkeypatch.setattr("voxreceipt.cli.load_adapter", load_failure)
    with pytest.raises(SystemExit) as exited:
        main(
            [
                "restore",
                "--input-dir",
                str(source),
                "--output-dir",
                str(output),
                "--report",
                str(report),
                "--adapter",
                "private-adapter",
                "--trust-adapter-code",
            ]
        )
    captured = capsys.readouterr()
    assert exited.value.code == 1
    assert captured.out == ""
    assert captured.err == "voxreceipt: adapter execution or output conversion failed\n"
    assert "Traceback" not in captured.err
    assert str(tmp_path) not in captured.err
    assert list(output.iterdir()) == []
    assert not report.exists()


def test_report_commit_failure_rolls_back_without_touching_replacement_directory(
    tmp_path: Path, sine: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    write_pcm16_mono(source / "same.wav", sine)
    output = tmp_path / "output"
    displaced = tmp_path / "displaced-output"
    report = tmp_path / "report.json"

    def replace_output_then_fail(self: object, payload: object) -> None:
        output.rename(displaced)
        output.mkdir()
        (output / "same.wav").write_bytes(b"replacement-directory-file")
        raise SafetyError("simulated report failure")

    monkeypatch.setattr("voxreceipt.safeio.PreparedJsonReport.commit", replace_output_then_fail)
    with pytest.raises(SafetyError, match="simulated report failure"):
        restore_directory(source, output, IdentityAdapter(), report_path=report)
    assert (output / "same.wav").read_bytes() == b"replacement-directory-file"
    assert list(displaced.iterdir()) == []
    assert not report.exists()


class OutputReplacingAdapter:
    name = "directory-replacement-test"
    version = "1"
    kind = "test-only"

    def __init__(self, output: Path, displaced: Path) -> None:
        self.output = output
        self.displaced = displaced

    def restore(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        self.output.rename(self.displaced)
        self.output.mkdir()
        (self.output / "same.wav").write_bytes(b"replacement-directory-file")
        return samples


def test_restore_detects_output_directory_replacement_and_cleans_original(
    tmp_path: Path, sine: np.ndarray
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    write_pcm16_mono(source / "same.wav", sine)
    output = tmp_path / "output"
    displaced = tmp_path / "displaced-output"
    report = tmp_path / "report.json"
    adapter = OutputReplacingAdapter(output, displaced)
    with pytest.raises(SafetyError, match="output directory changed"):
        restore_directory(source, output, adapter, report_path=report)
    assert (output / "same.wav").read_bytes() == b"replacement-directory-file"
    assert list(displaced.iterdir()) == []
    assert not report.exists()


@dataclass(frozen=True)
class LengthAdapter:
    difference: int
    name: str = "length-test"
    version: str = "1"
    kind: str = "test-only"

    def restore(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.difference >= 0:
            return np.pad(samples, (0, self.difference))
        return samples[: self.difference]


@pytest.mark.parametrize("difference", [-3_001, 3_001])
def test_restore_rejects_length_and_cleans_created_files(tmp_path: Path, difference: int) -> None:
    source = tmp_path / "input"
    source.mkdir()
    write_pcm16_mono(source / "first.wav", np.zeros(5_000))
    output = tmp_path / "output"
    with pytest.raises(AdapterError, match="3000"):
        restore_directory(source, output, LengthAdapter(difference))
    assert list(output.iterdir()) == []


@dataclass(frozen=True)
class NaNAdapter:
    name: str = "nan-test"
    version: str = "1"
    kind: str = "test-only"

    def restore(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.asarray([np.nan])


def test_restore_rejects_nan_and_leaves_no_audio(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    write_pcm16_mono(source / "x.wav", [0.0, 0.1])
    output = tmp_path / "output"
    with pytest.raises(AdapterError, match="invalid"):
        restore_directory(source, output, NaNAdapter())
    assert list(output.iterdir()) == []


def test_restore_content_gate_can_fail_closed(tmp_path: Path, sine: np.ndarray) -> None:
    source = tmp_path / "input"
    source.mkdir()
    write_pcm16_mono(source / "x.wav", sine)
    output = tmp_path / "output"
    with pytest.raises(AdapterError, match="content-proxy"):
        restore_directory(source, output, LengthAdapter(-100), minimum_content_proxy=1.0)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan])
def test_restore_content_threshold_bounds(tmp_path: Path, threshold: float) -> None:
    source = tmp_path / "input"
    source.mkdir()
    with pytest.raises(MetricError, match="between"):
        restore_directory(
            source, tmp_path / "output", IdentityAdapter(), minimum_content_proxy=threshold
        )


def test_restore_refuses_nonempty_output(tmp_path: Path) -> None:
    source, output = roots(tmp_path)
    write_pcm16_mono(source / "x.wav", [0.0])
    (output / "keep.txt").write_text("keep")
    with pytest.raises(SafetyError, match="empty"):
        restore_directory(source, output, IdentityAdapter())
    assert (output / "keep.txt").read_text() == "keep"


def test_evaluate_three_authorized_sets(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    degraded_dir = tmp_path / "degraded"
    restored_dir = tmp_path / "restored"
    for path in (clean_dir, degraded_dir, restored_dir):
        path.mkdir()
    clean = synthetic_speechlike(duration_seconds=0.1)
    degraded = degrade_signal(clean)
    write_pcm16_mono(clean_dir / "test.wav", clean)
    write_pcm16_mono(degraded_dir / "test.wav", degraded)
    write_pcm16_mono(restored_dir / "test.wav", clean)
    report = evaluate_directories(clean_dir, degraded_dir, restored_dir)
    assert report["summary"]["evaluated_count"] == 1
    assert report["summary"]["mean_si_sdr_improvement_db"] > 0
    assert "official dataset" in report["scope"]


def test_evaluate_requires_identical_sets(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    degraded_dir = tmp_path / "degraded"
    restored_dir = tmp_path / "restored"
    for path in (clean_dir, degraded_dir, restored_dir):
        path.mkdir()
    write_pcm16_mono(clean_dir / "only.wav", [0.0, 0.1])
    with pytest.raises(AudioFormatError, match="identical"):
        evaluate_directories(clean_dir, degraded_dir, restored_dir)


def test_demo_creates_complete_offline_artifacts(tmp_path: Path) -> None:
    summary = _make_demo(tmp_path / "demo")
    assert summary["preflight_valid"] is True
    assert summary["evaluated_count"] == 1
    assert (tmp_path / "demo" / "preflight-report.json").exists()
    assert (tmp_path / "demo" / "data-ledger.json").exists()
    restore = json.loads((tmp_path / "demo" / "restore-report.json").read_text())
    preflight = json.loads((tmp_path / "demo" / "preflight-report.json").read_text())
    evaluation = json.loads((tmp_path / "demo" / "evaluation-report.json").read_text())
    assert restore["files"][0]["file_id"] == preflight["files"][0]["file_id"]
    assert restore["files"][0]["file_id"] == evaluation["files"][0]["file_id"]


def test_demo_refuses_existing_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "demo"
    workspace.mkdir()
    with pytest.raises(Exception, match="already exists"):
        _make_demo(workspace)


def test_parser_requires_subcommand() -> None:
    with pytest.raises(SystemExit) as result:
        build_parser().parse_args([])
    assert result.value.code == 2


def test_main_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    assert "voxreceipt 0.1.0" in capsys.readouterr().out


def test_cli_preflight_returns_two_when_invalid(tmp_path: Path) -> None:
    source, output = roots(tmp_path)
    write_pcm16_mono(source / "missing.wav", [0.0])
    args = build_parser().parse_args(
        [
            "preflight",
            "--input-dir",
            str(source),
            "--output-dir",
            str(output),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    assert _run(args) == 2


def test_cli_ledger_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from test_safety_ledger import write_ledger

    ledger = tmp_path / "ledger.json"
    write_ledger(ledger)
    args = build_parser().parse_args(["ledger", "--file", str(ledger)])
    assert _run(args) == 0
    assert json.loads(capsys.readouterr().out)["dataset_count"] == 1


def test_adapter_is_protocol_at_runtime() -> None:
    assert isinstance(IdentityAdapter(), AudioRestorer)
