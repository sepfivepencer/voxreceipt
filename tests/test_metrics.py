from __future__ import annotations

import numpy as np
import pytest

from voxreceipt.errors import AudioFormatError, MetricError
from voxreceipt.metrics import (
    content_preservation,
    estimate_delay,
    latency_metrics,
    reference_metrics,
)
from voxreceipt.signals import degrade_signal, synthetic_speechlike


def test_identity_content_score_is_exact_one(sine: np.ndarray) -> None:
    report = content_preservation(sine, sine)
    assert report["proxy_score"] == 1.0
    assert report["estimated_delay_samples"] == 0
    assert "not proof" in str(report["interpretation"])


def test_exact_common_overlap_has_platform_stable_perfect_score() -> None:
    source = np.linspace(-0.271, 0.193, 4_801, dtype=np.float64)
    report = content_preservation(source, source[:-137], max_lag=0)
    assert report["waveform_correlation"] == 1.0
    assert report["envelope_correlation"] == 1.0
    assert report["spectral_cosine"] == 1.0
    assert report["proxy_score"] == 1.0


@pytest.mark.parametrize("delay", [-37, -1, 0, 1, 83])
def test_delay_estimation_sign(delay: int, sine: np.ndarray) -> None:
    if delay >= 0:
        candidate = np.pad(sine, (delay, 0))[: sine.size]
    else:
        candidate = np.pad(sine[-delay:], (0, -delay))
    assert estimate_delay(sine, candidate, 100) == delay


def test_content_metric_detects_polarity_inversion(sine: np.ndarray) -> None:
    report = content_preservation(sine, -sine)
    assert report["waveform_correlation"] == 0.0
    assert float(report["proxy_score"]) < 0.6


def test_content_metric_handles_silence() -> None:
    report = content_preservation(np.zeros(1000), np.zeros(1000))
    assert report["proxy_score"] == pytest.approx(1.0)
    assert report["estimated_delay_samples"] == 0
    assert np.isfinite(float(report["energy_ratio_db"]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_content_metric_rejects_non_finite(bad: float) -> None:
    with pytest.raises(MetricError, match="non-finite"):
        content_preservation([0.0, bad], [0.0, 0.0])


@pytest.mark.parametrize("values", [[], np.zeros((2, 2))])
def test_content_metric_rejects_bad_shape(values: object) -> None:
    with pytest.raises(MetricError):
        content_preservation(values, [0.0, 0.1])


def test_reference_metric_identity_improves_degraded_signal() -> None:
    clean = synthetic_speechlike(duration_seconds=0.2)
    degraded = degrade_signal(clean)
    report = reference_metrics(clean, degraded, clean)
    assert report["si_sdr_improvement_db"] > 0
    assert "not an official" in report["scope"]


def test_reference_metric_lengths_use_common_prefix() -> None:
    clean = np.linspace(-0.2, 0.2, 1000)
    report = reference_metrics(clean, clean[:900] + 0.01, clean[:800])
    assert np.isfinite(report["si_sdr_improvement_db"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.1])
def test_latency_rejects_bad_elapsed(bad: float) -> None:
    with pytest.raises(MetricError, match="elapsed"):
        latency_metrics(bad, 48_000)


@pytest.mark.parametrize("count", [0, -1])
def test_latency_rejects_nonpositive_duration(count: int) -> None:
    with pytest.raises(MetricError, match="positive"):
        latency_metrics(0.1, count)


def test_latency_real_time_factor() -> None:
    report = latency_metrics(0.25, 48_000)
    assert report == {"elapsed_ms": 250.0, "audio_seconds": 1.0, "real_time_factor": 0.25}


def test_negative_max_lag_rejected(sine: np.ndarray) -> None:
    with pytest.raises(MetricError, match="cannot be negative"):
        estimate_delay(sine, sine, -1)


@pytest.mark.parametrize("duration", [0.01, 31.0, np.nan])
def test_synthetic_duration_bounds(duration: float) -> None:
    with pytest.raises(AudioFormatError):
        synthetic_speechlike(duration_seconds=duration)


def test_synthetic_and_degradation_are_deterministic() -> None:
    first = synthetic_speechlike(duration_seconds=0.1, seed=99)
    second = synthetic_speechlike(duration_seconds=0.1, seed=99)
    assert np.array_equal(first, second)
    assert np.array_equal(degrade_signal(first, seed=4), degrade_signal(first, seed=4))


def test_degradation_preserves_length() -> None:
    clean = synthetic_speechlike(duration_seconds=0.1)
    assert degrade_signal(clean).size == clean.size
