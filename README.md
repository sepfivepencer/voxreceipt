# VoxReceipt

VoxReceipt checks a speech-restoration batch before you package it for the 2026 iFlytek
VoiceFixer Season 2 challenge. It runs offline, keeps audio on your machine, and writes a receipt
without filenames, local paths, or raw samples.

VoxReceipt is an alpha preparation tool. It is not an official scorer, a trained AI model, a
submission package, or evidence of a competition result.

## Why use it

The official rules can invalidate an entire submission for one missing, extra, unreadable, or
misformatted output. VoxReceipt checks the published hard constraints:

- the input and output filename sets match one to one;
- every output is a 48 kHz, PCM16, single-channel RIFF/WAVE file;
- each output differs from its input by at most 3000 samples;
- a local adapter returns finite one-dimensional audio;
- reports replace filenames with keyed HMAC identifiers.

VoxReceipt also measures adapter-only wall time and signal-level content proxies. Those proxy
values can flag a suspicious transformation. They cannot prove that a model preserved every word,
speaker trait, or meaning.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
voxreceipt demo --workspace demo-run
```

The demo generates one procedural, non-human signal, applies a small deterministic DSP baseline,
then writes restoration, preflight, evaluation, and license-ledger JSON files under `demo-run/`.
It downloads no audio or model.

Run the test suite from a source checkout:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src/voxreceipt
coverage run -m pytest
coverage report
```

## Current scope

| Included | Deferred until the participant supplies it |
|---|---|
| strict, bounded RIFF/WAVE parser | lawful training and validation data |
| batch restore adapter protocol | a participant-owned restoration model |
| format and file-set preflight | the organizer's hidden test set and scorer |
| latency and signal proxy metrics | NISQA, SIGMOS, DNSMOS, or an official MOS result |
| procedural smoke-test signal | the organizer's separate inference-package specification |
| dataset-rights ledger schema | S3 upload, registration, payment, and formal submission |

The `spectral-floor` baseline exists to exercise the pipeline. It uses deterministic spectral
subtraction and makes no AI or quality claim.

## Documentation

- [Build your first offline receipt](docs/tutorial-getting-started.md) (tutorial)
- [How to preflight a restoration batch](docs/how-to-preflight.md) (task guide)
- [CLI, Python API, schemas, and constraints](docs/reference.md) (reference)
- [Trust boundaries, privacy, and metric limits](docs/explanation-trust-boundaries.md)
  (explanation)
- [Contribution rules](CONTRIBUTING.md)

Each document is reachable from this page, and the four user documents link back to one another.

## Rules and prior work

The project checked the [official VoiceFixer page-data endpoint](https://challenge.xfyun.cn/2020/ai-contest/api/page-data/VoiceFixer)
on 2026-09-05. The endpoint states a 2026-09-24 17:00 submission deadline, the WAV constraints
above, a maximum of three submissions per team per day, no model API, and restrictions on sharing
original competition audio. Recheck the organizer's page before submission because the organizer
can revise rules.

The [original VoiceFixer repository](https://github.com/haoheliu/voicefixer) and
[VoiceFixer paper](https://arxiv.org/abs/2204.05841) implement neural speech restoration.
VoxReceipt does not copy or bundle that model. Its contribution is the combination of an offline
adapter boundary, fail-closed batch checks, pseudonymous receipts, proxy metrics, procedural test
signals, and a license ledger. The project does not claim that no one has combined similar ideas.

On 2026-09-05, exact-name searches covered GitHub repository names, the PyPI project endpoint, and
general web results. They found no exact `VoxReceipt` repository, Python package, or audio tool.
This check cannot replace a jurisdiction-specific trademark and company-name search.

## Privacy and security

VoxReceipt contains no network client. It reads direct-child WAV files from directories you name,
rejects symlinks, hard links, and unexpected entries, refuses report and audio overwrites, and keeps
reports outside audio directories. WAV input is capped at 64 MiB and 12,000,000 frames; the license
ledger is capped at 1,000,000 bytes. During `restore`, VoxReceipt reserves the report before calling
the adapter and rolls back WAVs created by that run if processing or report commit fails.

On POSIX systems, directory components are opened without following links and writes and rollback
stay bound to held directory descriptors. This prevents a renamed/replaced output-directory path
from redirecting cleanup into the replacement. The strongest descriptor guarantees depend on
`dir_fd` and `O_NOFOLLOW`; the continuous-integration security tests run on Linux. A custom adapter
is executable Python code, so the CLI imports one only when you pass `--trust-adapter-code`.

Set `VOXRECEIPT_ID_KEY` to exactly 64 hexadecimal characters if you need the same anonymous file ID
across commands. Without it, VoxReceipt creates an in-memory key and IDs remain stable only inside
one report or demo run. The key never appears in a report.
IDs authenticate the exact filename bytes produced by `os.fsencode`; Unicode NFC and NFD spellings
remain distinct, and POSIX surrogate-escaped non-UTF-8 names are supported. Consequently, unusual
non-ASCII names may not retain an ID when moved to a filesystem with different filename encoding.

Do not commit official audio, private speech, model credentials, generated reports, or the HMAC key.

## License

VoxReceipt is available under the [MIT License](LICENSE).
