"""Typed value objects shared across VoxReceipt modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WavInfo:
    """Header metadata for one WAV file."""

    sample_rate: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    compression: str

    @property
    def duration_seconds(self) -> float:
        """Return duration in seconds, or zero for an invalid zero rate."""

        return self.frame_count / self.sample_rate if self.sample_rate else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)
