"""Small, strict ledger for dataset license and competition-use decisions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from voxreceipt.constants import LEDGER_SCHEMA, MAX_LEDGER_BYTES
from voxreceipt.errors import LedgerError, SafetyError
from voxreceipt.safeio import DirectoryHandle

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_USES = {"training", "validation", "competition", "research", "redistribution"}
ENTRY_KEYS = {
    "dataset_id",
    "source_uri",
    "license_id",
    "allowed_uses",
    "contains_human_voice",
    "competition_use_confirmed",
    "evidence_sha256",
    "notes",
}


def _contains_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if _contains_surrogate(key):
            raise LedgerError("license ledger contains a surrogate code point")
        if key in result:
            raise LedgerError("license ledger contains a duplicate JSON key")
        result[key] = value
    return result


def _bounded_integer(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > 1_000:
        raise LedgerError("license ledger contains an oversized integer")
    try:
        return int(value)
    except ValueError:
        raise LedgerError("license ledger contains an invalid integer") from None


def _reject_surrogates(value: object) -> None:
    if isinstance(value, str):
        if _contains_surrogate(value):
            raise LedgerError("license ledger contains a surrogate code point")
        return
    if isinstance(value, list):
        for item in value:
            _reject_surrogates(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_surrogates(key)
            _reject_surrogates(item)


def _safe_text(value: object, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise LedgerError(f"{label} must be a bounded string")
    if any(ord(character) < 32 for character in value) or _contains_surrogate(value):
        raise LedgerError(f"{label} contains control characters")
    lowered = value.lower()
    if "file://" in lowered or "/users/" in lowered or "\\users\\" in lowered:
        raise LedgerError(f"{label} must not contain a local path")
    return value


def _validate_source(value: object) -> str:
    source = _safe_text(value, "source_uri", maximum=500)
    try:
        parsed = urlsplit(source)
    except ValueError:
        raise LedgerError("source_uri is invalid") from None
    if parsed.scheme == "generated":
        if not parsed.netloc or parsed.username or parsed.password:
            raise LedgerError("generated source URI is invalid")
    elif parsed.scheme == "https":
        if not parsed.netloc or parsed.username or parsed.password:
            raise LedgerError("HTTPS source URI is invalid")
    else:
        raise LedgerError("source_uri must use https:// or generated://")
    return source


def _validate_entry(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != ENTRY_KEYS:
        raise LedgerError("each dataset entry must use exactly the documented fields")
    dataset_id = value["dataset_id"]
    license_id = value["license_id"]
    if not isinstance(dataset_id, str) or not SAFE_ID.fullmatch(dataset_id):
        raise LedgerError("dataset_id is invalid")
    if not isinstance(license_id, str) or not SAFE_ID.fullmatch(license_id):
        raise LedgerError("license_id is invalid")
    uses = value["allowed_uses"]
    if (
        not isinstance(uses, list)
        or not uses
        or any(not isinstance(item, str) or item not in ALLOWED_USES for item in uses)
        or len(set(uses)) != len(uses)
    ):
        raise LedgerError("allowed_uses must be a unique non-empty list of known uses")
    for field in ("contains_human_voice", "competition_use_confirmed"):
        if not isinstance(value[field], bool):
            raise LedgerError(f"{field} must be boolean")
    evidence = value["evidence_sha256"]
    if not isinstance(evidence, str) or not SHA256.fullmatch(evidence):
        raise LedgerError("evidence_sha256 must be a lowercase SHA-256 digest")
    notes = _safe_text(value["notes"], "notes", maximum=500)
    return {
        "dataset_id": dataset_id,
        "source_uri": _validate_source(value["source_uri"]),
        "license_id": license_id,
        "allowed_uses": sorted(uses),
        "contains_human_voice": value["contains_human_voice"],
        "competition_use_confirmed": value["competition_use_confirmed"],
        "evidence_sha256": evidence,
        "notes": notes,
    }


def _read_ledger(path: Path) -> bytes:
    """Read one bounded, unchanged regular file through a single no-follow descriptor."""

    if path.name in {"", ".", ".."}:
        raise LedgerError("license ledger filename is invalid")
    fd: int | None = None
    try:
        with DirectoryHandle.open(path.parent) as directory:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path.name, flags, dir_fd=directory.fd)
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise LedgerError("license ledger must be a single-link regular file")
            if before.st_size > MAX_LEDGER_BYTES:
                raise LedgerError("license ledger exceeds one megabyte")

            remaining = before.st_size
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(fd, min(65_536, remaining))
                if not chunk:
                    raise LedgerError("license ledger changed while it was read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise LedgerError("license ledger grew while it was read")

            after = os.fstat(fd)
            try:
                current = os.stat(path.name, dir_fd=directory.fd, follow_symlinks=False)
            except OSError:
                raise LedgerError("license ledger was replaced while it was read") from None
            unchanged = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_nlink,
            ) == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            )
            same_name = (
                current.st_dev == after.st_dev
                and current.st_ino == after.st_ino
                and current.st_nlink == 1
                and stat.S_ISREG(current.st_mode)
            )
            if not unchanged or not same_name:
                raise LedgerError("license ledger was changed or replaced while it was read")
            return b"".join(chunks)
    except LedgerError:
        raise
    except (OSError, SafetyError) as exc:
        raise LedgerError("license ledger must be a readable single-link regular file") from exc
    finally:
        if fd is not None:
            os.close(fd)


def validate_ledger(path: Path) -> dict[str, Any]:
    """Validate a ledger and return a privacy-minimal digest summary."""

    try:
        raw = _read_ledger(path)
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_int=_bounded_integer,
        )
        _reject_surrogates(payload)
    except LedgerError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise LedgerError("license ledger is unreadable JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema", "datasets"}:
        raise LedgerError("license ledger root must contain schema and datasets")
    if payload["schema"] != LEDGER_SCHEMA or not isinstance(payload["datasets"], list):
        raise LedgerError("license ledger schema is unsupported")
    if not payload["datasets"]:
        raise LedgerError("license ledger must contain at least one dataset")
    entries = [_validate_entry(item) for item in payload["datasets"]]
    ids = [entry["dataset_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise LedgerError("dataset_id values must be unique")
    canonical = json.dumps(
        {"schema": LEDGER_SCHEMA, "datasets": entries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": LEDGER_SCHEMA,
        "dataset_count": len(entries),
        "competition_use_confirmed_count": sum(
            bool(entry["competition_use_confirmed"]) for entry in entries
        ),
        "contains_human_voice_count": sum(bool(entry["contains_human_voice"]) for entry in entries),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "legal_conclusion": "none; the participant remains responsible for rights and consent",
    }
