from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

import pytest

from voxreceipt.constants import LEDGER_SCHEMA, MAX_LEDGER_BYTES
from voxreceipt.errors import LedgerError, SafetyError
from voxreceipt.ledger import validate_ledger
from voxreceipt.privacy import Pseudonymizer, pseudonymizer_from_environment
from voxreceipt.safeio import (
    checked_directory,
    create_or_check_empty_directory,
    ensure_separate_directories,
    prepare_json_report,
    wav_files,
    write_json_exclusive,
)
from voxreceipt.wav import write_pcm16_mono


def valid_entry() -> dict[str, object]:
    return {
        "dataset_id": "procedural-v1",
        "source_uri": "generated://voxreceipt/procedural-v1",
        "license_id": "MIT",
        "allowed_uses": ["training", "competition"],
        "contains_human_voice": False,
        "competition_use_confirmed": True,
        "evidence_sha256": "a" * 64,
        "notes": "Generated signal without recorded speech.",
    }


def write_ledger(path: Path, entry: dict[str, object] | None = None) -> None:
    path.write_text(json.dumps({"schema": LEDGER_SCHEMA, "datasets": [entry or valid_entry()]}))


def test_valid_ledger_returns_minimal_summary(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    write_ledger(path)
    summary = validate_ledger(path)
    assert summary["dataset_count"] == 1
    assert summary["competition_use_confirmed_count"] == 1
    assert summary["contains_human_voice_count"] == 0
    assert len(str(summary["canonical_sha256"])) == 64
    assert "datasets" not in summary


def test_shipped_example_ledger_matches_license() -> None:
    import hashlib

    project = Path(__file__).parents[1]
    summary = validate_ledger(project / "examples" / "data-ledger.json")
    entry = json.loads((project / "examples" / "data-ledger.json").read_text())["datasets"][0]
    assert (
        entry["evidence_sha256"] == hashlib.sha256((project / "LICENSE").read_bytes()).hexdigest()
    )
    assert summary["contains_human_voice_count"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "bad id"),
        ("source_uri", "http://insecure.example/data"),
        ("source_uri", "file:///Users/alice/private.wav"),
        ("license_id", "bad license"),
        ("allowed_uses", []),
        ("allowed_uses", ["training", "training"]),
        ("allowed_uses", ["unknown"]),
        ("contains_human_voice", "false"),
        ("competition_use_confirmed", 1),
        ("evidence_sha256", "ABC"),
        ("notes", "/Users/alice/private.wav"),
        ("notes", "line\nbreak"),
    ],
)
def test_invalid_ledger_fields_rejected(tmp_path: Path, field: str, value: object) -> None:
    entry = valid_entry()
    entry[field] = value
    path = tmp_path / "bad.json"
    write_ledger(path, entry)
    with pytest.raises(LedgerError):
        validate_ledger(path)


def test_duplicate_json_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"voxreceipt.ledger.v1","schema":"x","datasets":[]}')
    with pytest.raises(LedgerError, match="duplicate"):
        validate_ledger(path)


def test_duplicate_dataset_id_rejected(tmp_path: Path) -> None:
    entry = valid_entry()
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps({"schema": LEDGER_SCHEMA, "datasets": [entry, entry]}))
    with pytest.raises(LedgerError, match="unique"):
        validate_ledger(path)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema": "wrong", "datasets": [valid_entry()]},
        {"schema": LEDGER_SCHEMA, "datasets": []},
        {"schema": LEDGER_SCHEMA, "datasets": "not-list"},
    ],
)
def test_invalid_ledger_root_rejected(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "root.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(LedgerError):
        validate_ledger(path)


def test_ledger_symlink_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    write_ledger(target)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(LedgerError, match="regular"):
        validate_ledger(link)


def test_ledger_hard_link_rejected(tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    linked = tmp_path / "linked.json"
    write_ledger(original)
    os.link(original, linked)
    with pytest.raises(LedgerError, match="single-link"):
        validate_ledger(linked)


def test_ledger_byte_limit_rejected_before_read(tmp_path: Path) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b"{")
    os.truncate(path, MAX_LEDGER_BYTES + 1)
    with pytest.raises(LedgerError, match="one megabyte"):
        validate_ledger(path)


def test_ledger_growth_during_bounded_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "growing.json"
    write_ledger(path)
    original_read = os.read
    first = True

    def grow_after_read(fd: int, size: int) -> bytes:
        nonlocal first
        chunk = original_read(fd, size)
        if first:
            first = False
            with path.open("ab") as handle:
                handle.write(b" ")
        return chunk

    monkeypatch.setattr("voxreceipt.ledger.os.read", grow_after_read)
    with pytest.raises(LedgerError, match="grew"):
        validate_ledger(path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX rename/unlink semantics")
def test_ledger_replacement_during_single_fd_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "replace.json"
    write_ledger(path)
    original_read = os.read
    first = True

    def replace_after_read(fd: int, size: int) -> bytes:
        nonlocal first
        chunk = original_read(fd, size)
        if first:
            first = False
            path.unlink()
            write_ledger(path)
        return chunk

    monkeypatch.setattr("voxreceipt.ledger.os.read", replace_after_read)
    with pytest.raises(LedgerError, match="changed or replaced"):
        validate_ledger(path)


def test_ledger_oversized_integer_is_safely_rejected(tmp_path: Path) -> None:
    path = tmp_path / "integer.json"
    path.write_text('{"schema":' + "9" * 5_000 + ',"datasets":[]}')
    with pytest.raises(LedgerError, match="oversized integer"):
        validate_ledger(path)


@pytest.mark.parametrize(
    "raw",
    [
        '{"sche\\ud800ma":"x","datasets":[]}',
        '{"schema":"voxreceipt.ledger.v1","datasets":[{"bad\\udfff":"x"}]}',
        (
            '{"schema":"voxreceipt.ledger.v1","datasets":[{'
            '"dataset_id":"x","source_uri":"generated://x/y",'
            '"license_id":"MIT","allowed_uses":["competition"],'
            '"contains_human_voice":false,"competition_use_confirmed":true,'
            '"evidence_sha256":"' + "a" * 64 + '","notes":"bad\\ud800"}]}'
        ),
    ],
)
def test_ledger_rejects_surrogates_in_keys_and_fields(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "surrogate.json"
    path.write_text(raw)
    with pytest.raises(LedgerError, match="surrogate"):
        validate_ledger(path)


def test_ledger_malformed_uri_is_safely_rejected(tmp_path: Path) -> None:
    entry = valid_entry()
    entry["source_uri"] = "https://["
    path = tmp_path / "uri.json"
    write_ledger(path, entry)
    with pytest.raises(LedgerError, match="source_uri"):
        validate_ledger(path)


def test_json_report_exclusive_and_private(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    write_json_exclusive(report, {"ok": True})
    assert json.loads(report.read_text()) == {"ok": True}
    assert os.stat(report).st_mode & 0o777 == 0o600
    with pytest.raises(SafetyError, match="already exists"):
        write_json_exclusive(report, {"ok": False})


def test_report_symlink_is_not_followed(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("keep")
    link = tmp_path / "report.json"
    link.symlink_to(target)
    with pytest.raises(SafetyError, match="already exists"):
        write_json_exclusive(link, {"bad": True})
    assert target.read_text() == "keep"


def test_failed_report_serialization_removes_reservation(tmp_path: Path) -> None:
    report = tmp_path / "bad-report.json"
    with pytest.raises(SafetyError, match="written safely"):
        write_json_exclusive(report, {"bad": "\ud800"})
    assert not report.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX unlink-open-file semantics")
def test_report_commit_rejects_replacement_without_deleting_it(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    prepared = prepare_json_report(report_path)
    report_path.unlink()
    report_path.write_text("replacement")
    try:
        with pytest.raises(SafetyError, match="written safely"):
            prepared.commit({"ok": True})
    finally:
        prepared.close()
    assert report_path.read_text() == "replacement"


def test_checked_directory_rejects_link(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(SafetyError, match="not a link"):
        checked_directory(link)


def test_checked_directory_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    child = real_parent / "child"
    child.mkdir(parents=True)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(SafetyError, match="link"):
        checked_directory(linked_parent / "child")


def test_create_or_check_empty_directory(tmp_path: Path) -> None:
    created = create_or_check_empty_directory(tmp_path / "new")
    assert created.is_dir()
    (created / "file").write_text("x")
    with pytest.raises(SafetyError, match="empty"):
        create_or_check_empty_directory(created)


@pytest.mark.parametrize("nested", ["same", "child", "parent"])
def test_separate_directories_reject_overlap(tmp_path: Path, nested: str) -> None:
    left = tmp_path / "left"
    left.mkdir()
    child = left / "child"
    child.mkdir()
    right = left if nested == "same" else child
    if nested == "parent":
        left, right = child, left
    with pytest.raises(SafetyError, match="separate"):
        ensure_separate_directories(left, right)


@pytest.mark.parametrize("entry_kind", ["text", "directory", "symlink"])
def test_wav_directory_rejects_unexpected_entry(tmp_path: Path, entry_kind: str) -> None:
    root = tmp_path / "audio"
    root.mkdir()
    if entry_kind == "text":
        (root / "notes.txt").write_text("x")
    elif entry_kind == "directory":
        (root / "nested.wav").mkdir()
    else:
        target = tmp_path / "target.wav"
        write_pcm16_mono(target, [0.0])
        (root / "link.wav").symlink_to(target)
    with pytest.raises(SafetyError, match="only"):
        wav_files(root)


def test_pseudonym_is_keyed_stable_and_preserves_exact_name_bytes() -> None:
    first = Pseudonymizer(b"a" * 32, persistent=True)
    second = Pseudonymizer(b"b" * 32, persistent=True)
    assert first.identify("caf\N{LATIN SMALL LETTER E WITH ACUTE}.wav") != first.identify(
        "cafe\u0301.wav"
    )
    assert first.identify("voice.wav") == first.identify("voice.wav")
    assert first.identify("voice.wav") != second.identify("voice.wav")
    assert "voice" not in first.identify("voice.wav")
    expected = hmac.new(
        b"a" * 32,
        b"voxreceipt-filename-v1\0" + os.fsencode("voice.wav"),
        hashlib.sha256,
    ).hexdigest()[:32]
    assert first.identify("voice.wav") == f"wav_{expected}"


@pytest.mark.skipif(os.name != "posix", reason="POSIX surrogateescape filename behavior")
def test_pseudonym_accepts_non_utf8_posix_filename_bytes() -> None:
    filename = os.fsdecode(b"voice-\xff.wav")
    identifier = Pseudonymizer(b"a" * 32, persistent=True).identify(filename)
    assert identifier.startswith("wav_")
    assert len(identifier) == 36


@pytest.mark.parametrize("encoded", ["bad", "00", "gg" * 32])
def test_environment_key_validation(monkeypatch: pytest.MonkeyPatch, encoded: str) -> None:
    monkeypatch.setenv("VOXRECEIPT_ID_KEY", encoded)
    with pytest.raises(SafetyError):
        pseudonymizer_from_environment()


def test_environment_key_produces_persistent_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXRECEIPT_ID_KEY", "12" * 32)
    first = pseudonymizer_from_environment()
    second = pseudonymizer_from_environment()
    assert first.persistent is True
    assert first.identify("x.wav") == second.identify("x.wav")


def test_missing_environment_key_is_ephemeral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXRECEIPT_ID_KEY", raising=False)
    assert pseudonymizer_from_environment().persistent is False
