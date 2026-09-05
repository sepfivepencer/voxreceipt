# Build your first offline restoration receipt

You will generate a synthetic test signal, restore it with the bundled DSP baseline, and inspect a
submission preflight report. The fixture contains no recorded voice and needs no network access.

## What you need

- Python 3.10 or later
- a source checkout of VoxReceipt
- a shell that can activate a Python virtual environment

## Step 1: Install VoxReceipt

From the repository root, create an isolated environment and install the package:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

Confirm the installed command:

```bash
voxreceipt --version
```

Expected output starts with `voxreceipt 0.1.0`.

## Step 2: Create a visible result

Run the self-contained demo into a new directory:

```bash
voxreceipt demo --workspace demo-run
```

The command prints a one-line JSON summary. A successful run includes
`"preflight_valid": true` and `"evaluated_count": 1`.

The new directory contains:

```text
demo-run/
├── clean/procedural.wav
├── degraded/procedural.wav
├── restored/procedural.wav
├── data-ledger.json
├── restore-report.json
├── preflight-report.json
└── evaluation-report.json
```

The three WAV files come from equations and seeded noise in `signals.py`. No person spoke into a
microphone to create them.

## Step 3: Read the hard-gate result

Open `demo-run/preflight-report.json`. Its summary has this shape:

```json
{
  "checked_count": 1,
  "failure_count": 0,
  "input_count": 1,
  "output_count": 1,
  "valid": true
}
```

The file record shows a `file_id`, header metadata, and signed sample delta. It omits the filename
and all local paths. The demo passes one pseudonym key to each stage, so the same `file_id` appears
in all three reports.

## Step 4: Separate published rules from proxy measurements

Compare the reports:

- `preflight-report.json` checks the published set, format, and sample-count requirements.
- `restore-report.json` records adapter metadata, adapter-only wall time, and input-to-output signal
  proxies.
- `evaluation-report.json` calculates SI-SDR against the procedural clean reference.

Only the first report checks requirements copied from the public competition page. The content
proxy and SI-SDR are local diagnostics. Neither reproduces the organizer's MOS scoring stack.

## Step 5: Confirm the dataset ledger

Run the ledger validator:

```bash
voxreceipt ledger --file demo-run/data-ledger.json
```

The summary reports one dataset, no human voice, and one competition-use confirmation. The digest
identifies the canonical ledger content. It does not grant legal permission. You must verify rights
for any dataset that you add.

## What you built

You now have a complete offline smoke test for the adapter, WAV writer, pseudonymous receipt, and
preflight gate. Use the [batch preflight guide](how-to-preflight.md) for your own local model. Read
the [reference](reference.md) for every flag and report field, then review the
[trust-boundary explanation](explanation-trust-boundaries.md) before using private audio.
