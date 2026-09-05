"""Batch restoration, format preflight, and paired offline evaluation."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from voxreceipt.adapters import AudioRestorer, validate_adapter
from voxreceipt.constants import (
    CHANNELS,
    EVALUATION_SCHEMA,
    MAX_SAMPLE_DELTA,
    OFFICIAL_RULES_URL,
    PREFLIGHT_SCHEMA,
    RESTORE_SCHEMA,
    RULES_CHECKED_DATE,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
)
from voxreceipt.errors import AdapterError, AudioFormatError, MetricError, SafetyError
from voxreceipt.ledger import validate_ledger
from voxreceipt.metrics import content_preservation, latency_metrics, reference_metrics
from voxreceipt.privacy import Pseudonymizer, pseudonymizer_from_environment
from voxreceipt.safeio import (
    CreatedFile,
    DirectoryHandle,
    PreparedJsonReport,
    checked_directory,
    ensure_separate_directories,
    open_or_create_empty_directory,
    prepare_json_report,
    remove_created_files,
    wav_files,
    write_json_exclusive,
)
from voxreceipt.wav import inspect_wav, read_pcm16_mono, write_pcm16_mono_at


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rules() -> dict[str, Any]:
    return {
        "source": OFFICIAL_RULES_URL,
        "checked_date": RULES_CHECKED_DATE,
        "required_sample_rate": SAMPLE_RATE,
        "required_sample_width_bytes": SAMPLE_WIDTH_BYTES,
        "required_channels": CHANNELS,
        "maximum_absolute_sample_delta": MAX_SAMPLE_DELTA,
        "required_file_mapping": "same filenames, exactly one output per input, no extras",
    }


def _privacy(pseudonyms: Pseudonymizer) -> dict[str, Any]:
    return {
        "contains_audio": False,
        "contains_filenames": False,
        "contains_absolute_paths": False,
        "file_id_method": (
            "domain-separated HMAC-SHA256 over exact filesystem filename bytes, "
            "truncated to 128 bits; key is never written"
        ),
        "file_id_stability": "cross-run" if pseudonyms.persistent else "this-report-only",
    }


def _maybe_write(path: Path | None, payload: dict[str, Any]) -> None:
    if path is not None:
        write_json_exclusive(path, payload)


def _validate_report_location(report_path: Path | None, audio_roots: tuple[Path, ...]) -> None:
    if report_path is None:
        return
    try:
        with DirectoryHandle.open(report_path.parent) as report_directory:
            report_parent = report_directory.path
            report_identity = (report_directory.device, report_directory.inode)
        for root in audio_roots:
            with DirectoryHandle.open(root) as audio_directory:
                if (
                    report_identity == (audio_directory.device, audio_directory.inode)
                    or audio_directory.path == report_parent
                    or audio_directory.path in report_parent.parents
                ):
                    raise SafetyError("report cannot be stored inside an audio directory")
    except SafetyError:
        raise


def preflight_directories(
    input_dir: Path,
    output_dir: Path,
    *,
    report_path: Path | None = None,
    pseudonyms: Pseudonymizer | None = None,
) -> dict[str, Any]:
    """Check the official file-set, WAV-format, and sample-length constraints."""

    source_root = checked_directory(input_dir)
    result_root = checked_directory(output_dir)
    ensure_separate_directories(source_root, result_root)
    _validate_report_location(report_path, (source_root, result_root))
    source_files = wav_files(source_root)
    result_files = wav_files(result_root)
    ids = pseudonyms or pseudonymizer_from_environment()
    records: list[dict[str, Any]] = []
    passed = True
    for name in sorted(set(source_files) | set(result_files)):
        record: dict[str, Any] = {"file_id": ids.identify(name), "errors": []}
        source = source_files.get(name)
        result = result_files.get(name)
        if source is None:
            record["errors"].append("unexpected_output")
        if result is None:
            record["errors"].append("missing_output")
        source_info = None
        result_info = None
        if source is not None:
            try:
                source_info = inspect_wav(source, require_official_format=True)
                record["input"] = source_info.to_dict()
            except AudioFormatError:
                record["errors"].append("input_format_or_container_invalid")
        if result is not None:
            try:
                result_info = inspect_wav(result, require_official_format=True)
                record["output"] = result_info.to_dict()
            except AudioFormatError:
                record["errors"].append("output_format_or_container_invalid")
        if source_info is not None and result_info is not None:
            delta = result_info.frame_count - source_info.frame_count
            record["sample_delta"] = delta
            record["absolute_sample_delta"] = abs(delta)
            if abs(delta) > MAX_SAMPLE_DELTA:
                record["errors"].append("sample_delta_exceeds_3000")
        record["valid"] = not record["errors"]
        passed = passed and bool(record["valid"])
        records.append(record)
    if not source_files:
        passed = False
    payload: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "generated_at": _timestamp(),
        "rules": _rules(),
        "privacy": _privacy(ids),
        "summary": {
            "valid": passed,
            "input_count": len(source_files),
            "output_count": len(result_files),
            "checked_count": len(records),
            "failure_count": sum(not bool(record["valid"]) for record in records),
        },
        "files": records,
    }
    _maybe_write(report_path, payload)
    return payload


def restore_directory(
    input_dir: Path,
    output_dir: Path,
    adapter: AudioRestorer,
    *,
    report_path: Path | None = None,
    ledger_path: Path | None = None,
    minimum_content_proxy: float | None = None,
    pseudonyms: Pseudonymizer | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one trusted adapter over a directory and create a privacy-minimal receipt."""

    if minimum_content_proxy is not None and not 0.0 <= minimum_content_proxy <= 1.0:
        raise MetricError("minimum content proxy must be between zero and one")
    source_root = checked_directory(input_dir)
    with open_or_create_empty_directory(output_dir) as result_directory:
        result_root = result_directory.path
        ensure_separate_directories(source_root, result_root)
        _validate_report_location(report_path, (source_root, result_root))
        source_files = wav_files(source_root)
        if not source_files:
            raise AudioFormatError("input directory contains no WAV files")
        trusted_adapter = validate_adapter(adapter)
        ledger_summary = validate_ledger(ledger_path) if ledger_path is not None else None
        ids = pseudonyms or pseudonymizer_from_environment()
        created: list[CreatedFile] = []
        records: list[dict[str, Any]] = []
        prepared_report: PreparedJsonReport | None = None
        try:
            # Reservation is deliberately before the first adapter call. Report serialization and
            # durable commit remain inside the same rollback boundary as every generated WAV.
            if report_path is not None:
                prepared_report = prepare_json_report(report_path)
            for name, source in source_files.items():
                original = read_pcm16_mono(source)
                started = clock()
                try:
                    restored = trusted_adapter.restore(original.copy(), SAMPLE_RATE)
                    candidate = np.asarray(restored, dtype=np.float64)
                except Exception:
                    raise AdapterError("adapter execution or output conversion failed") from None
                elapsed = clock() - started
                if candidate.ndim != 1 or not np.all(np.isfinite(candidate)):
                    raise AdapterError("adapter returned invalid audio samples")
                delta = int(candidate.size - original.size)
                if abs(delta) > MAX_SAMPLE_DELTA:
                    raise AdapterError("adapter output violates the 3000-sample length limit")
                content = content_preservation(original, candidate)
                if (
                    minimum_content_proxy is not None
                    and float(content["proxy_score"]) < minimum_content_proxy
                ):
                    raise AdapterError("adapter output failed the configured content-proxy gate")
                created.append(write_pcm16_mono_at(result_directory, name, candidate))
                records.append(
                    {
                        "file_id": ids.identify(name),
                        "input_frames": int(original.size),
                        "output_frames": int(candidate.size),
                        "sample_delta": delta,
                        "latency": latency_metrics(elapsed, int(original.size)),
                        "content_proxy": content,
                    }
                )
            payload: dict[str, Any] = {
                "schema": RESTORE_SCHEMA,
                "generated_at": _timestamp(),
                "rules": _rules(),
                "privacy": _privacy(ids),
                "adapter": {
                    "name": trusted_adapter.name,
                    "version": trusted_adapter.version,
                    "kind": trusted_adapter.kind,
                },
                "content_gate": {
                    "minimum_proxy": minimum_content_proxy,
                    "limitation": (
                        "signal proxy only; it cannot prove words or speaker identity are unchanged"
                    ),
                },
                "license_ledger": ledger_summary,
                "summary": {"valid": True, "processed_count": len(records)},
                "files": records,
            }
            if not result_directory.still_names_path():
                raise SafetyError("output directory changed during restoration")
            if prepared_report is not None:
                prepared_report.commit(payload)
            return payload
        except BaseException:
            remove_created_files(created, result_directory)
            raise
        finally:
            if prepared_report is not None:
                prepared_report.close()


def evaluate_directories(
    clean_dir: Path,
    degraded_dir: Path,
    restored_dir: Path,
    *,
    report_path: Path | None = None,
    pseudonyms: Pseudonymizer | None = None,
) -> dict[str, Any]:
    """Evaluate same-named, authorized clean/degraded/restored WAV triples offline."""

    clean_root = checked_directory(clean_dir)
    degraded_root = checked_directory(degraded_dir)
    restored_root = checked_directory(restored_dir)
    ensure_separate_directories(clean_root, degraded_root, restored_root)
    _validate_report_location(report_path, (clean_root, degraded_root, restored_root))
    clean_files = wav_files(clean_root)
    degraded_files = wav_files(degraded_root)
    restored_files = wav_files(restored_root)
    names = set(clean_files)
    if not names or names != set(degraded_files) or names != set(restored_files):
        raise AudioFormatError("evaluation directories must contain identical non-empty WAV sets")
    ids = pseudonyms or pseudonymizer_from_environment()
    records: list[dict[str, Any]] = []
    for name in sorted(names):
        clean = read_pcm16_mono(clean_files[name])
        degraded = read_pcm16_mono(degraded_files[name])
        restored = read_pcm16_mono(restored_files[name])
        records.append(
            {"file_id": ids.identify(name), "metrics": reference_metrics(clean, degraded, restored)}
        )
    improvements = [float(record["metrics"]["si_sdr_improvement_db"]) for record in records]
    payload: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "generated_at": _timestamp(),
        "privacy": _privacy(ids),
        "scope": "offline paired audio supplied by the user; no official dataset or scorer",
        "summary": {
            "evaluated_count": len(records),
            "mean_si_sdr_improvement_db": float(np.mean(improvements)),
        },
        "files": records,
    }
    _maybe_write(report_path, payload)
    return payload
