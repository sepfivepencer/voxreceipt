"""Offline proxy metrics for timing and content-preservation risk."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from voxreceipt.constants import MAX_SAMPLE_DELTA, SAMPLE_RATE
from voxreceipt.errors import MetricError

FloatArray = npt.NDArray[np.float64]


def _samples(values: npt.ArrayLike, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise MetricError(f"{label} must be non-empty one-dimensional audio")
    if not np.all(np.isfinite(result)):
        raise MetricError(f"{label} contains non-finite samples")
    return result


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def estimate_delay(reference: npt.ArrayLike, candidate: npt.ArrayLike, max_lag: int) -> int:
    """Estimate candidate delay using FFT cross-correlation, bounded to ``max_lag``."""

    left = _samples(reference, "reference")
    right = _samples(candidate, "candidate")
    if max_lag < 0:
        raise MetricError("maximum lag cannot be negative")
    if np.linalg.norm(left) <= 1e-15 or np.linalg.norm(right) <= 1e-15:
        return 0
    fft_size = _next_power_of_two(left.size + right.size - 1)
    correlation = np.fft.irfft(
        np.fft.rfft(right, fft_size) * np.conj(np.fft.rfft(left, fft_size)), fft_size
    )
    bound = min(max_lag, max(left.size, right.size) - 1)
    lags = np.arange(-bound, bound + 1, dtype=np.int64)
    indices = np.where(lags >= 0, lags, fft_size + lags)
    # Pick the strongest relationship by magnitude, then let the signed
    # waveform metric penalize polarity inversion. Maximizing only positive
    # correlation can misread an inverted periodic signal as a time shift.
    return int(lags[int(np.argmax(np.abs(correlation[indices])))])


def _aligned(
    reference: FloatArray, candidate: FloatArray, delay: int
) -> tuple[FloatArray, FloatArray]:
    if delay >= 0:
        length = min(reference.size, candidate.size - delay)
        left = reference[: max(0, length)]
        right = candidate[delay : delay + max(0, length)]
    else:
        offset = -delay
        length = min(reference.size - offset, candidate.size)
        left = reference[offset : offset + max(0, length)]
        right = candidate[: max(0, length)]
    if left.size == 0:
        raise MetricError("audio has no overlap after delay alignment")
    return left, right


def _correlation(left: FloatArray, right: FloatArray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-15:
        return 1.0 if np.allclose(left, right, atol=1e-12) else 0.0
    return float(np.clip(np.dot(left_centered, right_centered) / denominator, -1.0, 1.0))


def _rms_envelope(values: FloatArray, frame_size: int = 960, hop: int = 480) -> FloatArray:
    if values.size <= frame_size:
        return np.asarray([np.sqrt(np.mean(np.square(values)))], dtype=np.float64)
    starts = range(0, values.size - frame_size + 1, hop)
    return np.asarray(
        [np.sqrt(np.mean(np.square(values[start : start + frame_size]))) for start in starts],
        dtype=np.float64,
    )


def _spectral_cosine(left: FloatArray, right: FloatArray) -> float:
    size = min(left.size, right.size, 65_536)
    window = np.hanning(size) if size > 1 else np.ones(size)
    left_spectrum = np.log1p(np.abs(np.fft.rfft(left[:size] * window)))
    right_spectrum = np.log1p(np.abs(np.fft.rfft(right[:size] * window)))
    denominator = float(np.linalg.norm(left_spectrum) * np.linalg.norm(right_spectrum))
    if denominator <= 1e-15:
        return 1.0 if np.allclose(left[:size], right[:size], atol=1e-12) else 0.0
    return float(np.clip(np.dot(left_spectrum, right_spectrum) / denominator, 0.0, 1.0))


def content_preservation(
    original: npt.ArrayLike,
    restored: npt.ArrayLike,
    *,
    max_lag: int = MAX_SAMPLE_DELTA,
) -> dict[str, float | int | str]:
    """Measure signal-level similarity; this cannot prove semantic preservation."""

    source = _samples(original, "original")
    output = _samples(restored, "restored")
    delay = estimate_delay(source, output, max_lag)
    aligned_source, aligned_output = _aligned(source, output, delay)
    if np.array_equal(aligned_source, aligned_output):
        # Exact equality has an exact score. Taking the dot/norm/FFT path can produce either 1.0
        # or 1.0 minus one ULP across otherwise supported NumPy wheels and CPU backends.
        waveform = envelope = spectral = score = 1.0
    else:
        waveform = max(0.0, _correlation(aligned_source, aligned_output))
        source_envelope = _rms_envelope(aligned_source)
        output_envelope = _rms_envelope(aligned_output)
        envelope = max(0.0, _correlation(source_envelope, output_envelope))
        spectral = _spectral_cosine(aligned_source, aligned_output)
        score = float(np.clip(0.5 * waveform + 0.3 * envelope + 0.2 * spectral, 0.0, 1.0))
    source_rms = float(np.sqrt(np.mean(np.square(aligned_source))))
    output_rms = float(np.sqrt(np.mean(np.square(aligned_output))))
    energy_ratio_db = float(20.0 * np.log10((output_rms + 1e-12) / (source_rms + 1e-12)))
    result: dict[str, float | int | str] = {
        "estimated_delay_samples": delay,
        "waveform_correlation": waveform,
        "envelope_correlation": envelope,
        "spectral_cosine": spectral,
        "energy_ratio_db": energy_ratio_db,
        "proxy_score": score,
        "interpretation": (
            "signal-level proxy; not proof that words or speaker identity are preserved"
        ),
    }
    if not all(np.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise MetricError("content metric produced a non-finite value")
    return result


def _si_sdr(reference: FloatArray, candidate: FloatArray) -> float:
    target = reference - np.mean(reference)
    estimate = candidate - np.mean(candidate)
    target_energy = float(np.dot(target, target))
    if target_energy <= 1e-15:
        return 120.0 if np.allclose(target, estimate, atol=1e-12) else -120.0
    projection = np.dot(estimate, target) / target_energy * target
    residual = estimate - projection
    ratio = float(np.dot(projection, projection)) / max(float(np.dot(residual, residual)), 1e-12)
    return float(np.clip(10.0 * np.log10(max(ratio, 1e-12)), -120.0, 120.0))


def reference_metrics(
    clean: npt.ArrayLike,
    degraded: npt.ArrayLike,
    restored: npt.ArrayLike,
) -> dict[str, Any]:
    """Compare authorized paired signals and report SI-SDR change plus proxy retention."""

    clean_values = _samples(clean, "clean")
    degraded_values = _samples(degraded, "degraded")
    restored_values = _samples(restored, "restored")
    common = min(clean_values.size, degraded_values.size, restored_values.size)
    if common < 2:
        raise MetricError("paired signals are too short")
    clean_values = clean_values[:common]
    degraded_values = degraded_values[:common]
    restored_values = restored_values[:common]
    before = _si_sdr(clean_values, degraded_values)
    after = _si_sdr(clean_values, restored_values)
    improvement = float(np.clip(after - before, -240.0, 240.0))
    if not np.isfinite(improvement):
        raise MetricError("reference metric produced a non-finite value")
    return {
        "si_sdr_degraded_db": before,
        "si_sdr_restored_db": after,
        "si_sdr_improvement_db": improvement,
        "content_proxy": content_preservation(degraded_values, restored_values),
        "scope": "paired authorized audio only; not an official MOS score",
    }


def latency_metrics(
    elapsed_seconds: float, sample_count: int, sample_rate: int = SAMPLE_RATE
) -> dict[str, float]:
    """Return wall-clock latency and real-time factor with finite-value checks."""

    if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise MetricError("elapsed time must be finite and non-negative")
    if sample_count <= 0 or sample_rate <= 0:
        raise MetricError("latency measurement needs positive audio duration")
    duration = sample_count / sample_rate
    return {
        "elapsed_ms": float(elapsed_seconds * 1000.0),
        "audio_seconds": float(duration),
        "real_time_factor": float(elapsed_seconds / duration),
    }
