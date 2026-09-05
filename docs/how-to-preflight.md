# How to preflight a speech-restoration batch

This guide runs a trusted local adapter, creates outputs without replacement, and checks the
published VoiceFixer file constraints.

## Prerequisites

- Python 3.10 or later with VoxReceipt installed
- a directory containing direct-child `.wav` inputs
- permission to process every audio file and use the associated model
- a model adapter installed in the same local Python environment

Do not put reports inside the input or output directory. The official output directory may contain
only the matching WAV set.

## 1. Inspect one input

Check the strict competition format before running a model:

```bash
voxreceipt inspect --file authorized-input/example.wav --official-format
```

The command exits with code 1 if the RIFF container or 48 kHz, PCM16, mono header fails validation.

## 2. Record dataset rights

Copy `examples/data-ledger.json` to a private working location. Replace its sample entry with one
entry for each data source. The `evidence_sha256` should identify the license, consent record, or
other rights evidence you reviewed. Do not put a local path, person's name, credential, or raw
audio in the ledger.

Validate the result:

```bash
voxreceipt ledger --file private-work/data-ledger.json
```

`competition_use_confirmed_count` should match the sources that you checked for this competition.
The validator records your decision; it does not make the legal decision for you.

## 3. Implement the adapter contract

Place your adapter in an installed local package. The object needs safe metadata and one method:

```python
import numpy as np
import numpy.typing as npt


class MyRestorer:
    name = "team-model"
    version = "2026.09"
    kind = "local-model"

    def restore(
        self,
        samples: npt.NDArray[np.float64],
        sample_rate: int,
    ) -> npt.NDArray[np.float64]:
        if sample_rate != 48_000:
            raise ValueError("expected 48 kHz")
        # This runnable contract example is an identity operation. Replace this
        # line with your inspected local model call, then keep the same checks.
        result = samples.copy()
        return np.asarray(result, dtype=np.float64)


adapter = MyRestorer()
```

VoxReceipt passes float64 mono samples in memory. Your adapter must return a finite one-dimensional
array. The hard sample-length gate accepts signed deltas from -3000 through +3000.

Custom adapters run Python code with your user permissions. Install and inspect the module before
you grant trust.

## 4. Choose stable anonymous IDs

VoxReceipt creates fresh in-memory HMAC keys by default. Use one 32-byte key when you need the same
anonymous file IDs across restore, preflight, and evaluation runs.

Generate 64 hexadecimal characters with a password manager or an operating-system random generator.
Load them without placing the value in a command-line argument:

```bash
read -r -s VOXRECEIPT_ID_KEY
export VOXRECEIPT_ID_KEY
```

Paste the 64 characters at the hidden prompt. Keep the value out of source control and shared logs.

## 5. Run the local adapter

Choose an output directory that does not exist or exists as an empty real directory:

```bash
voxreceipt restore \
  --input-dir authorized-input \
  --output-dir restored-output \
  --adapter my_team.adapter:adapter \
  --trust-adapter-code \
  --ledger private-work/data-ledger.json \
  --minimum-content-proxy 0.60 \
  --report private-work/restore-report.json
```

VoxReceipt measures only the call to `restore()`. It excludes WAV read and write time. The optional
content threshold can stop a batch when the signal proxy drops below your chosen value. Test and
document your threshold because the proxy cannot verify words or speaker identity.

VoxReceipt removes files created by the current run if the adapter fails before the batch finishes.
It never replaces a pre-existing output.

## 6. Run the published-constraint gate

```bash
voxreceipt preflight \
  --input-dir authorized-input \
  --output-dir restored-output \
  --report private-work/preflight-report.json
```

Exit code 0 means every checked public constraint passed. Exit code 2 means the report contains at
least one missing output, unexpected output, invalid container or format, or sample delta outside
the inclusive range.

## 7. Evaluate authorized clean references

Skip this step when you lack a lawful clean reference. For paired data you may process, run:

```bash
voxreceipt evaluate \
  --clean-dir authorized-clean \
  --degraded-dir authorized-input \
  --restored-dir restored-output \
  --report private-work/evaluation-report.json
```

The three directories must contain the same non-empty filename set. VoxReceipt writes SI-SDR change
and content proxies. The organizer scores hidden data with a MOS toolchain, so do not label these
numbers as leaderboard estimates.

## Verification

Check the preflight summary:

```bash
python -c 'import json; print(json.load(open("private-work/preflight-report.json"))["summary"])'
```

Confirm `valid` is `True`, `failure_count` is `0`, and both file counts match. Re-run the organizer's
current packaging instructions before creating `tar.gz`; VoxReceipt does not certify that archive
layout or upload it to S3.

## Troubleshooting

### Output directory must be empty

Move the old directory to an archive location or choose a new destination. VoxReceipt will not
delete or replace existing output files.

### Audio directory may contain only WAV files

Remove reports, hidden metadata files, nested directories, and links from the batch directory. Keep
only single-link regular direct-child `.wav` files. Re-copy a hard-linked file so it has its own inode.
For conservative submission compatibility, ensure each RIFF file places `fmt` before `data`; unknown
odd-sized chunks are allowed only when their padding byte is present. Files above 64 MiB or
12,000,000 frames exceed VoxReceipt's local resource ceiling.

### A failed restore left a reserved file after interruption

Normal Python failures, including report-write failures, roll back WAVs from that run and remove the
uncommitted report reservation. A process kill or machine crash cannot execute rollback. Inspect the
destination and report target, then move them aside or remove them under your normal recovery policy
before retrying; VoxReceipt will never overwrite them.

### Custom adapters require explicit trust

Pass `--trust-adapter-code` only after you inspect and install that local module. The flag does not
sandbox the adapter.

### The content-proxy gate rejected an output

Inspect the audio under your approved privacy process and compare multiple content checks. Reduce
the threshold only after you document why the proxy produced a false alarm.

See the [reference](reference.md) for exact schemas and the
[trust-boundary explanation](explanation-trust-boundaries.md) for privacy limits.
