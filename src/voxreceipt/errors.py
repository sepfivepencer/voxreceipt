"""Stable, non-sensitive error types."""


class VoxReceiptError(Exception):
    """Base error suitable for presentation by the CLI."""


class AudioFormatError(VoxReceiptError):
    """A WAV file does not meet a required format constraint."""


class SafetyError(VoxReceiptError):
    """A path or write operation failed a safety check."""


class LedgerError(VoxReceiptError):
    """A data-license ledger is invalid."""


class AdapterError(VoxReceiptError):
    """An adapter cannot be loaded or returned invalid samples."""


class MetricError(VoxReceiptError):
    """A metric received invalid or insufficient samples."""
