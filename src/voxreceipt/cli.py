"""Command-line entry point for fully offline VoxReceipt workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

from voxreceipt import __version__
from voxreceipt.adapters import DspBaseline, load_adapter
from voxreceipt.constants import LEDGER_SCHEMA
from voxreceipt.errors import VoxReceiptError
from voxreceipt.ledger import validate_ledger
from voxreceipt.preflight import evaluate_directories, preflight_directories, restore_directory
from voxreceipt.privacy import pseudonymizer_from_environment
from voxreceipt.safeio import checked_directory, write_json_exclusive
from voxreceipt.signals import degrade_signal, synthetic_speechlike
from voxreceipt.wav import inspect_wav, write_pcm16_mono


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxreceipt",
        description="Offline speech-restoration evaluation and submission preflight",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="build and check a procedural offline fixture")
    demo.add_argument("--workspace", type=_path, required=True)

    restore = commands.add_parser("restore", help="run a local in-memory adapter")
    restore.add_argument("--input-dir", type=_path, required=True)
    restore.add_argument("--output-dir", type=_path, required=True)
    restore.add_argument("--report", type=_path, required=True)
    restore.add_argument("--adapter", default="identity")
    restore.add_argument("--trust-adapter-code", action="store_true")
    restore.add_argument("--ledger", type=_path)
    restore.add_argument("--minimum-content-proxy", type=float)

    preflight = commands.add_parser("preflight", help="validate official output constraints")
    preflight.add_argument("--input-dir", type=_path, required=True)
    preflight.add_argument("--output-dir", type=_path, required=True)
    preflight.add_argument("--report", type=_path, required=True)

    evaluate = commands.add_parser("evaluate", help="score authorized paired offline audio")
    evaluate.add_argument("--clean-dir", type=_path, required=True)
    evaluate.add_argument("--degraded-dir", type=_path, required=True)
    evaluate.add_argument("--restored-dir", type=_path, required=True)
    evaluate.add_argument("--report", type=_path, required=True)

    ledger = commands.add_parser("ledger", help="validate a dataset-rights ledger")
    ledger.add_argument("--file", type=_path, required=True)

    inspect = commands.add_parser(
        "inspect", help="inspect one WAV without reading it into a report"
    )
    inspect.add_argument("--file", type=_path, required=True)
    inspect.add_argument("--official-format", action="store_true")
    return parser


def _make_demo(workspace: Path) -> dict[str, object]:
    if workspace.exists() or workspace.is_symlink():
        raise VoxReceiptError("demo workspace already exists; refusing to overwrite")
    parent = checked_directory(workspace.parent)
    root = parent / workspace.name
    try:
        root.mkdir(mode=0o700)
        clean_dir = root / "clean"
        degraded_dir = root / "degraded"
        output_dir = root / "restored"
        clean_dir.mkdir(mode=0o700)
        degraded_dir.mkdir(mode=0o700)
    except OSError as exc:
        raise VoxReceiptError("demo workspace could not be created") from exc

    clean = synthetic_speechlike(duration_seconds=0.6)
    degraded = degrade_signal(clean)
    write_pcm16_mono(clean_dir / "procedural.wav", clean)
    write_pcm16_mono(degraded_dir / "procedural.wav", degraded)
    ledger_payload = {
        "schema": LEDGER_SCHEMA,
        "datasets": [
            {
                "dataset_id": "voxreceipt-procedural-v1",
                "source_uri": "generated://voxreceipt/procedural-v1",
                "license_id": "MIT",
                "allowed_uses": [
                    "training",
                    "validation",
                    "competition",
                    "research",
                    "redistribution",
                ],
                "contains_human_voice": False,
                "competition_use_confirmed": True,
                "evidence_sha256": (
                    "a8f91d866796de733bf566b4c436afbcc22fdce831a039bba90c3695712d8c32"
                ),
                "notes": "Procedurally generated test signal; no recording or human voice is used.",
            }
        ],
    }
    ledger_path = root / "data-ledger.json"
    write_json_exclusive(ledger_path, ledger_payload)
    pseudonyms = pseudonymizer_from_environment()
    restore_report = root / "restore-report.json"
    restore_directory(
        degraded_dir,
        output_dir,
        DspBaseline(),
        report_path=restore_report,
        ledger_path=ledger_path,
        pseudonyms=pseudonyms,
    )
    preflight = preflight_directories(
        degraded_dir,
        output_dir,
        report_path=root / "preflight-report.json",
        pseudonyms=pseudonyms,
    )
    evaluation = evaluate_directories(
        clean_dir,
        degraded_dir,
        output_dir,
        report_path=root / "evaluation-report.json",
        pseudonyms=pseudonyms,
    )
    return {
        "workspace_created": True,
        "fixture": "procedural non-human signal",
        "preflight_valid": preflight["summary"]["valid"],
        "evaluated_count": evaluation["summary"]["evaluated_count"],
    }


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _run(args: argparse.Namespace) -> int:
    if args.command == "demo":
        _print(_make_demo(args.workspace))
        return 0
    if args.command == "restore":
        adapter = load_adapter(args.adapter, trust_custom_code=args.trust_adapter_code)
        payload = restore_directory(
            args.input_dir,
            args.output_dir,
            adapter,
            report_path=args.report,
            ledger_path=args.ledger,
            minimum_content_proxy=args.minimum_content_proxy,
        )
        _print(payload["summary"])
        return 0
    if args.command == "preflight":
        payload = preflight_directories(args.input_dir, args.output_dir, report_path=args.report)
        _print(payload["summary"])
        return 0 if bool(payload["summary"]["valid"]) else 2
    if args.command == "evaluate":
        payload = evaluate_directories(
            args.clean_dir,
            args.degraded_dir,
            args.restored_dir,
            report_path=args.report,
        )
        _print(payload["summary"])
        return 0
    if args.command == "ledger":
        _print(validate_ledger(args.file))
        return 0
    if args.command == "inspect":
        _print(inspect_wav(args.file, require_official_format=args.official_format).to_dict())
        return 0
    raise VoxReceiptError("unknown command")


def main(argv: list[str] | None = None) -> NoReturn:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = _run(args)
    except VoxReceiptError as exc:
        print(f"voxreceipt: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
