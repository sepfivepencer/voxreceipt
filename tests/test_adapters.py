from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from voxreceipt.adapters import DspBaseline, IdentityAdapter, load_adapter, validate_adapter
from voxreceipt.errors import AdapterError
from voxreceipt.signals import degrade_signal, synthetic_speechlike


def test_identity_returns_copy(sine: np.ndarray) -> None:
    result = IdentityAdapter().restore(sine, 48_000)
    assert np.array_equal(result, sine)
    assert result is not sine


def test_identity_rejects_invalid_rate(sine: np.ndarray) -> None:
    with pytest.raises(AdapterError, match="sample rate"):
        IdentityAdapter().restore(sine, 0)


@pytest.mark.parametrize("strength", [-0.1, 2.1])
def test_dsp_rejects_strength(strength: float) -> None:
    with pytest.raises(AdapterError, match="strength"):
        DspBaseline(strength=strength)


@pytest.mark.parametrize("floor", [-0.1, 1.1])
def test_dsp_rejects_floor(floor: float) -> None:
    with pytest.raises(AdapterError, match="floor"):
        DspBaseline(floor=floor)


@pytest.mark.parametrize("size", [1, 100, 959, 960, 961, 4_801])
def test_dsp_preserves_length_and_finiteness(size: int) -> None:
    values = np.linspace(-0.2, 0.2, size)
    result = DspBaseline().restore(values, 48_000)
    assert result.shape == values.shape
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result)) <= 1.0


def test_dsp_is_deterministic() -> None:
    degraded = degrade_signal(synthetic_speechlike(duration_seconds=0.1))
    first = DspBaseline().restore(degraded, 48_000)
    second = DspBaseline().restore(degraded, 48_000)
    assert np.array_equal(first, second)


def test_dsp_handles_empty() -> None:
    assert DspBaseline().restore(np.asarray([]), 48_000).size == 0


@pytest.mark.parametrize("bad", [np.asarray([[0.0]]), np.asarray([np.nan])])
def test_dsp_rejects_invalid_samples(bad: np.ndarray) -> None:
    with pytest.raises(AdapterError, match="finite one-dimensional"):
        DspBaseline().restore(bad, 48_000)


def test_dsp_rejects_non_48k(sine: np.ndarray) -> None:
    with pytest.raises(AdapterError, match="48 kHz"):
        DspBaseline().restore(sine, 44_100)


@pytest.mark.parametrize("spec", ["identity", "dsp"])
def test_load_builtin(spec: str) -> None:
    assert load_adapter(spec).name


@pytest.mark.parametrize("spec", ["unknown", "os:path:extra", "bad-name:thing", "os:bad-name"])
def test_custom_adapter_requires_trust_or_safe_reference(spec: str) -> None:
    with pytest.raises(AdapterError):
        load_adapter(spec, trust_custom_code=spec != "unknown")


def test_custom_adapter_loads_only_after_explicit_trust(tmp_path: Path) -> None:
    module = tmp_path / "custom_adapter.py"
    module.write_text(
        "from voxreceipt.adapters import IdentityAdapter\n"
        "adapter = IdentityAdapter(name='custom-local')\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(AdapterError, match="explicit"):
            load_adapter("custom_adapter:adapter")
        assert load_adapter("custom_adapter:adapter", trust_custom_code=True).name == "custom-local"
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("custom_adapter", None)


@pytest.mark.parametrize(
    ("field", "value"),
    [("name", "space here"), ("name", "x" * 81), ("version", ""), ("kind", "secret/path")],
)
def test_adapter_metadata_is_sanitized(field: str, value: str) -> None:
    adapter = IdentityAdapter(**{field: value})
    with pytest.raises(AdapterError, match="metadata"):
        validate_adapter(adapter)


def test_non_adapter_rejected() -> None:
    with pytest.raises(AdapterError, match="protocol"):
        validate_adapter(object())
