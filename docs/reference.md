# VoxReceipt reference

VoxReceipt exposes six CLI commands and a typed Python API. Version 0.1.0 accepts Python 3.10 or
later and depends on NumPy 1.24 through 2.2.

## Command line

### Global options

```text
voxreceipt [-h] [--version] {demo,restore,preflight,evaluate,ledger,inspect} ...
```

| Option | Effect |
|---|---|
| `-h`, `--help` | Print help and exit 0 |
| `--version` | Print the installed version and exit 0 |

### `demo`

```text
voxreceipt demo --workspace PATH
```

Creates `PATH` with procedural clean, degraded, and restored WAV files plus four JSON artifacts.
`PATH` must not exist. The command uses the built-in `spectral-floor` adapter.

### `restore`

```text
voxreceipt restore --input-dir PATH --output-dir PATH --report PATH
                   [--adapter SPEC] [--trust-adapter-code]
                   [--ledger PATH] [--minimum-content-proxy FLOAT]
```

| Option | Type and default | Constraint |
|---|---|---|
| `--input-dir` | path, required | Real directory with regular direct-child WAV files |
| `--output-dir` | path, required | Missing or empty real directory, separate from all inputs |
| `--report` | path, required | New file outside every audio directory |
| `--adapter` | string, `identity` | `identity`, `dsp`, or `module:object` |
| `--trust-adapter-code` | flag, false | Required for `module:object`; grants normal Python process access |
| `--ledger` | path, absent | Optional ledger that must pass `voxreceipt.ledger.v1` validation |
| `--minimum-content-proxy` | float, absent | Inclusive 0 through 1; aborts below the threshold |

Built-in `identity` copies input samples. Built-in `dsp` selects `DspBaseline` with strength 0.7
and gain floor 0.15. Both exist for harness checks and carry non-AI `kind` metadata.

### `preflight`

```text
voxreceipt preflight --input-dir PATH --output-dir PATH --report PATH
```

Checks exact filename-set equality, strict WAV format, and signed frame-count delta. A report with
`summary.valid=false` still writes successfully and returns exit code 2.

### `evaluate`

```text
voxreceipt evaluate --clean-dir PATH --degraded-dir PATH
                    --restored-dir PATH --report PATH
```

Requires the same non-empty direct-child filename set in all three directories. All inputs must use
48 kHz PCM16 mono. The output contains per-file SI-SDR change and signal proxies.

### `ledger`

```text
voxreceipt ledger --file PATH
```

Validates a dataset-rights ledger and prints counts plus a canonical SHA-256 digest. It does not
print ledger entries.

### `inspect`

```text
voxreceipt inspect --file PATH [--official-format]
```

Prints sample rate, channels, sample width, frame count, compression, and duration. The optional
flag enforces the current public competition format.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Command completed; preflight found no violation |
| 1 | VoxReceipt rejected input, a path, an adapter, a metric, or a ledger |
| 2 | `preflight` wrote a report with at least one violation; `argparse` also uses 2 for bad syntax |

## Python API

VoxReceipt exports these names from `voxreceipt`:

```python
from voxreceipt import (
    AudioRestorer,
    DspBaseline,
    IdentityAdapter,
    WavInfo,
    content_preservation,
    evaluate_directories,
    inspect_wav,
    load_adapter,
    preflight_directories,
    read_pcm16_mono,
    reference_metrics,
    restore_directory,
    validate_ledger,
    write_pcm16_mono,
)
```

### Adapter protocol

```python
class AudioRestorer(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def kind(self) -> str: ...

    def restore(
        self,
        samples: numpy.ndarray,
        sample_rate: int,
    ) -> numpy.ndarray: ...
```

Each metadata value must match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$`. `restore()` receives a copy of
the input float64 array and 48000 as the sample rate. It must return a finite one-dimensional array
with an absolute length difference no larger than 3000 samples.

### Functions

```python
inspect_wav(path: Path, *, require_official_format: bool = False) -> WavInfo
read_pcm16_mono(path: Path, *, sample_rate: int = 48_000) -> NDArray[float64]
write_pcm16_mono(path: Path, samples: ArrayLike, *, sample_rate: int = 48_000) -> None
load_adapter(spec: str, *, trust_custom_code: bool = False) -> AudioRestorer
validate_ledger(path: Path) -> dict[str, Any]
content_preservation(original: ArrayLike, restored: ArrayLike, *, max_lag: int = 3_000) -> dict
reference_metrics(clean: ArrayLike, degraded: ArrayLike, restored: ArrayLike) -> dict
preflight_directories(
    input_dir: Path,
    output_dir: Path,
    *,
    report_path: Path | None = None,
    pseudonyms: Pseudonymizer | None = None,
) -> dict[str, Any]
restore_directory(
    input_dir: Path,
    output_dir: Path,
    adapter: AudioRestorer,
    *,
    report_path: Path | None = None,
    ledger_path: Path | None = None,
    minimum_content_proxy: float | None = None,
    pseudonyms: Pseudonymizer | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]
evaluate_directories(
    clean_dir: Path,
    degraded_dir: Path,
    restored_dir: Path,
    *,
    report_path: Path | None = None,
    pseudonyms: Pseudonymizer | None = None,
) -> dict[str, Any]
```

Import `Pseudonymizer` from `voxreceipt.privacy` when you call these options. Most callers can omit
it and let the process read `VOXRECEIPT_ID_KEY` or create an ephemeral key.

`write_pcm16_mono` clips finite input values to the PCM16 range and creates mode `0600` files. It
refuses an existing path. The RIFF parser requires `fmt` before `data` and skips bounded unknown
chunks with correct odd-byte padding. It rejects size mismatches, truncated chunks, duplicate `fmt`
or `data` chunks, inconsistent format fields, and trailing bytes outside RIFF.

## Public hard constraints

| Constant | Value |
|---|---:|
| sample rate | 48000 Hz |
| sample width | 2 bytes, signed PCM16 |
| channels | 1 |
| maximum absolute sample delta | 3000 |
| input/output mapping | exact same filename set |

`MAX_SAMPLE_DELTA` treats -3000 and +3000 as valid boundaries. It rejects -3001 and +3001.

These local resource ceilings are defensive implementation limits, not organizer rules:

| Constant | Value | Enforcement point |
|---|---:|---|
| `MAX_WAV_BYTES` | 67,108,864 bytes (64 MiB) | immediately after opening, before RIFF parsing |
| `MAX_WAV_FRAMES` | 12,000,000 | after bounded header parsing, before sample allocation |
| `MAX_LEDGER_BYTES` | 1,000,000 bytes | after `fstat`, before the bounded read loop |

WAV inputs and ledgers must be regular files with exactly one hard link. This deliberately excludes
hard-linked batch entries even when their bytes are otherwise valid.

## Report schemas

### Common privacy block

Every report declares that it contains no audio, filenames, or absolute paths. `file_id` is
`wav_` plus 32 lowercase hexadecimal characters from a keyed HMAC-SHA256 digest. The key does not
appear in JSON. The HMAC input is a domain separator plus the exact `os.fsencode(filename)` bytes;
VoxReceipt does not normalize Unicode. NFC and NFD names therefore receive different IDs, and POSIX
surrogate-escaped non-UTF-8 filenames are accepted.

`file_id_stability` has two values:

- `cross-run`: `VOXRECEIPT_ID_KEY` supplied the same 32-byte key;
- `this-report-only`: the process generated an in-memory key.

### `voxreceipt.preflight.v1`

Top-level fields: `schema`, `generated_at`, `rules`, `privacy`, `summary`, `files`.

Each file record contains `file_id`, `errors`, `valid`, available input/output header dictionaries,
and sample deltas when both headers pass. Error codes are:

- `unexpected_output`
- `missing_output`
- `input_format_or_container_invalid`
- `output_format_or_container_invalid`
- `sample_delta_exceeds_3000`

### `voxreceipt.restore.v1`

Top-level fields add `adapter`, `content_gate`, and `license_ledger`. Each file record contains frame
counts, signed sample delta, latency, and content proxies.

Latency fields:

| Field | Definition |
|---|---|
| `elapsed_ms` | wall time around the adapter call, in milliseconds |
| `audio_seconds` | input frames divided by 48000 |
| `real_time_factor` | elapsed seconds divided by audio seconds |

Content fields:

| Field | Range or unit |
|---|---|
| `estimated_delay_samples` | integer within configured lag window |
| `waveform_correlation` | 0 through 1 after negative values clamp to 0 |
| `envelope_correlation` | 0 through 1 after negative values clamp to 0 |
| `spectral_cosine` | 0 through 1 |
| `energy_ratio_db` | decibels |
| `proxy_score` | `0.5*waveform + 0.3*envelope + 0.2*spectral` |

### `voxreceipt.evaluation.v1`

Each record contains degraded and restored SI-SDR against the supplied clean reference, their
difference, and the degraded-to-restored content proxy. VoxReceipt caps reported SI-SDR to the
finite range -120 through +120 dB.

### `voxreceipt.ledger.v1`

The root has `schema` and a non-empty `datasets` list. Each dataset uses these exact fields:

| Field | Constraint |
|---|---|
| `dataset_id` | 1 to 80 safe identifier characters |
| `source_uri` | credential-free `https://` or `generated://` URI, at most 500 characters |
| `license_id` | 1 to 80 safe identifier characters |
| `allowed_uses` | unique non-empty subset of training, validation, competition, research, redistribution |
| `contains_human_voice` | JSON boolean |
| `competition_use_confirmed` | JSON boolean |
| `evidence_sha256` | 64 lowercase hexadecimal characters |
| `notes` | 1 to 500 characters, no controls or recognizable local user path |

VoxReceipt limits the ledger file to one megabyte and rejects hard links, changes or replacement
during its single-descriptor bounded read, oversized integers, surrogate code points in every key or
string field, duplicate JSON keys, and duplicate dataset IDs.

## Errors and safety behavior

Public functions raise subclasses of `VoxReceiptError`: `AudioFormatError`, `SafetyError`,
`LedgerError`, `AdapterError`, and `MetricError`. CLI errors omit local paths and file contents.

The batch APIs reject symlink roots or path components, nested roots, nested entries, hard-linked or
special files, and non-WAV entries. JSON and WAV writers use exclusive creation and request no-follow
behavior from the operating system. On POSIX, processing and rollback retain directory descriptors;
the implementation does not re-resolve the output pathname when deleting files created by a failed
run. `restore` reserves its report before calling the adapter and treats report commit as part of the
WAV rollback transaction. Ordinary adapter and NumPy conversion failures become a path-free
`AdapterError`.

Continue with the [preflight task guide](how-to-preflight.md), or read the
[trust-boundary explanation](explanation-trust-boundaries.md). The
[tutorial](tutorial-getting-started.md) provides a procedural first run.
