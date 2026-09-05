"""In-memory restoration adapter contract and deterministic reference baselines."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from voxreceipt.constants import SAMPLE_RATE
from voxreceipt.errors import AdapterError

FloatArray = npt.NDArray[np.float64]
SAFE_METADATA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
SAFE_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
SAFE_OBJECT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@runtime_checkable
class AudioRestorer(Protocol):
    """Protocol implemented by trusted, locally installed restoration adapters."""

    @property
    def name(self) -> str:
        """Safe adapter identifier included in the receipt."""

    @property
    def version(self) -> str:
        """Safe adapter version included in the receipt."""

    @property
    def kind(self) -> str:
        """Adapter category, such as deterministic-dsp-non-ai."""

    def restore(self, samples: FloatArray, sample_rate: int) -> FloatArray:
        """Return one-dimensional restored samples without file-system access."""


@dataclass(frozen=True)
class IdentityAdapter:
    """No-op adapter used to verify the harness, not to improve speech."""

    name: str = "identity"
    version: str = "1"
    kind: str = "test-only-non-ai"

    def restore(self, samples: FloatArray, sample_rate: int) -> FloatArray:
        if sample_rate <= 0:
            raise AdapterError("adapter received an invalid sample rate")
        return np.asarray(samples, dtype=np.float64).copy()


@dataclass(frozen=True)
class DspBaseline:
    """Light spectral subtraction baseline for harness demonstrations only."""

    strength: float = 0.7
    floor: float = 0.15
    name: str = "spectral-floor"
    version: str = "1"
    kind: str = "deterministic-dsp-non-ai"

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 2.0:
            raise AdapterError("DSP strength must be between 0 and 2")
        if not 0.0 <= self.floor <= 1.0:
            raise AdapterError("DSP gain floor must be between 0 and 1")

    def restore(self, samples: FloatArray, sample_rate: int) -> FloatArray:
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise AdapterError("adapter input must be finite one-dimensional audio")
        if sample_rate != SAMPLE_RATE:
            raise AdapterError("reference DSP accepts only 48 kHz audio")
        if values.size == 0:
            return values.copy()

        frame_size = 960
        hop = 480
        frame_count = max(1, 1 + int(np.ceil(max(0, values.size - frame_size) / hop)))
        padded_size = (frame_count - 1) * hop + frame_size
        padded = np.pad(values, (0, padded_size - values.size))
        window = np.sqrt(np.hanning(frame_size) + 1e-12)
        frames = np.stack(
            [
                padded[index * hop : index * hop + frame_size] * window
                for index in range(frame_count)
            ]
        )
        spectrum = np.fft.rfft(frames, axis=1)
        power = np.square(np.abs(spectrum))
        noise_power = np.percentile(power, 20.0, axis=0)
        gain = 1.0 - self.strength * noise_power[None, :] / (power + 1e-12)
        gain = np.clip(gain, self.floor, 1.0)
        restored_frames = np.fft.irfft(spectrum * gain, n=frame_size, axis=1) * window

        restored = np.zeros(padded_size, dtype=np.float64)
        normalization = np.zeros(padded_size, dtype=np.float64)
        for index, frame in enumerate(restored_frames):
            start = index * hop
            restored[start : start + frame_size] += frame
            normalization[start : start + frame_size] += np.square(window)
        restored /= np.maximum(normalization, 1e-8)
        clipped: FloatArray = np.asarray(
            np.clip(restored[: values.size], -1.0, 1.0), dtype=np.float64
        )
        if not np.all(np.isfinite(clipped)):
            raise AdapterError("reference DSP produced non-finite samples")
        return clipped


def validate_adapter(adapter: object) -> AudioRestorer:
    """Validate metadata and callable surface before any audio is processed."""

    if not isinstance(adapter, AudioRestorer):
        raise AdapterError("adapter does not implement the AudioRestorer protocol")
    for value in (adapter.name, adapter.version, adapter.kind):
        if not isinstance(value, str) or not SAFE_METADATA.fullmatch(value):
            raise AdapterError("adapter metadata must use safe printable identifiers")
    return adapter


def load_adapter(spec: str, *, trust_custom_code: bool = False) -> AudioRestorer:
    """Load a built-in adapter or an explicitly trusted ``module:object`` adapter."""

    if spec == "identity":
        return IdentityAdapter()
    if spec == "dsp":
        return DspBaseline()
    if not trust_custom_code:
        raise AdapterError("custom adapters require explicit --trust-adapter-code")
    if spec.count(":") != 1:
        raise AdapterError("custom adapter must be written as module:object")
    module_name, object_name = spec.split(":", maxsplit=1)
    if not SAFE_MODULE.fullmatch(module_name) or not SAFE_OBJECT.fullmatch(object_name):
        raise AdapterError("custom adapter reference is invalid")
    try:
        module = importlib.import_module(module_name)
        candidate = getattr(module, object_name)
        adapter = candidate() if isinstance(candidate, type) else candidate
    except Exception as exc:
        raise AdapterError("trusted custom adapter could not be loaded") from exc
    return validate_adapter(cast(object, adapter))
