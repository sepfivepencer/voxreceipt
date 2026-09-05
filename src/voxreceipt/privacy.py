"""Keyed filename pseudonyms for reports that do not expose local names."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass

from voxreceipt.errors import SafetyError

KEY_ENVIRONMENT_VARIABLE = "VOXRECEIPT_ID_KEY"


@dataclass(frozen=True)
class Pseudonymizer:
    """Produce stable opaque IDs for one secret key."""

    key: bytes
    persistent: bool

    def __post_init__(self) -> None:
        if len(self.key) < 32:
            raise SafetyError("pseudonym key must contain at least 32 bytes")

    def identify(self, filename: str) -> str:
        # A filename is an opaque filesystem identifier, not natural-language text. fsencode()
        # preserves POSIX surrogate-escaped bytes and deliberately keeps NFC and NFD distinct.
        exact_name = os.fsencode(filename)
        message = b"voxreceipt-filename-v1\0" + exact_name
        digest = hmac.new(self.key, message, hashlib.sha256).hexdigest()[:32]
        return f"wav_{digest}"


def pseudonymizer_from_environment() -> Pseudonymizer:
    """Use a 64-hex environment key, or an ephemeral random key if absent."""

    encoded = os.environ.get(KEY_ENVIRONMENT_VARIABLE)
    if encoded is None:
        return Pseudonymizer(secrets.token_bytes(32), persistent=False)
    try:
        key = bytes.fromhex(encoded)
    except ValueError as exc:
        raise SafetyError(f"{KEY_ENVIRONMENT_VARIABLE} must be hexadecimal") from exc
    if len(key) != 32:
        raise SafetyError(f"{KEY_ENVIRONMENT_VARIABLE} must encode exactly 32 bytes")
    return Pseudonymizer(key, persistent=True)
