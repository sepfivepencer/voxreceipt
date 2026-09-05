"""Procedural, non-human signals for offline smoke tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from voxreceipt.constants import SAMPLE_RATE
from voxreceipt.errors import AudioFormatError

FloatArray = npt.NDArray[np.float64]


def synthetic_speechlike(
    *, duration_seconds: float = 1.0, sample_rate: int = SAMPLE_RATE, seed: int = 7
) -> FloatArray:
    """Create a deterministic harmonic/noise test signal that contains no human voice."""

    if not np.isfinite(duration_seconds) or not 0.05 <= duration_seconds <= 30.0:
        raise AudioFormatError("synthetic duration must be between 0.05 and 30 seconds")
    if sample_rate != SAMPLE_RATE:
        raise AudioFormatError("synthetic fixture supports only 48 kHz")
    count = round(duration_seconds * sample_rate)
    time = np.arange(count, dtype=np.float64) / sample_rate
    fundamental = 128.0 + 18.0 * np.sin(2.0 * np.pi * 1.7 * time)
    phase = 2.0 * np.pi * np.cumsum(fundamental) / sample_rate
    voiced = sum(np.sin(index * phase) / index for index in range(1, 8))
    syllables = np.square(np.sin(np.pi * 3.4 * time))
    syllables *= 0.35 + 0.65 * np.square(np.sin(np.pi * 0.71 * time + 0.3))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(count)
    fricative = noise - np.convolve(noise, np.ones(17) / 17.0, mode="same")
    gate = (np.sin(2.0 * np.pi * 1.1 * time + 0.8) > 0.72).astype(np.float64)
    signal = 0.23 * voiced * syllables + 0.025 * fricative * gate
    fade = min(count // 2, int(sample_rate * 0.02))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=True)
        signal[:fade] *= ramp
        signal[-fade:] *= ramp[::-1]
    return np.asarray(np.clip(signal, -0.9, 0.9), dtype=np.float64)


def degrade_signal(clean: npt.ArrayLike, *, seed: int = 11) -> FloatArray:
    """Apply deterministic noise, echo, bandwidth loss, and mild clipping."""

    values = np.asarray(clean, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise AudioFormatError("clean synthetic signal must be finite one-dimensional audio")
    rng = np.random.default_rng(seed)
    noisy = values + 0.035 * rng.standard_normal(values.size)
    delay = min(1_920, max(1, values.size // 4))
    echoed = noisy.copy()
    echoed[delay:] += 0.28 * noisy[:-delay]
    kernel = np.hanning(15)
    kernel /= np.sum(kernel)
    band_limited = np.convolve(echoed, kernel, mode="same")
    return np.clip(band_limited, -0.42, 0.42).astype(np.float64)
