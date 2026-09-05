# Trust boundaries, privacy, and metric limits

Speech restoration mixes untrusted binary files, executable model code, and audio that may identify
a person. VoxReceipt separates those risks so one JSON receipt does not become another copy of a
private dataset.

## The problem

A well-formed WAV header can still lie about chunk sizes, and a batch can contain links that point
outside the chosen directory. A model can produce NaN values, replace speech content, or change the
length enough to fail platform ingestion. A helpful report can leak a speaker's name through the
filename or reveal a private directory structure.

No local tool can resolve every risk. Python adapter code runs with the caller's permissions, signal
correlation cannot understand words, and a ledger cannot decide whether a license applies to a
specific team. VoxReceipt treats each one as a separate trust decision.

## Processing boundary

```text
untrusted WAV directory
        |
        v
bounded RIFF parser  -> reject links, malformed chunks, wrong format
        |
        v
float64 sample copy
        |
        v
trusted local adapter  <- custom import requires --trust-adapter-code
        |
        v
finite/shape/length gate -> proxy content and adapter-only timing
        |
        v
exclusive PCM16 writer -> empty output directory, no replacement
        |
        +----> pseudonymous JSON receipt outside the audio directory
```

The parser reads chunk headers with bounded seeks. It compares the declared RIFF length with the
opened regular file, checks every chunk boundary before seeking, and validates PCM format arithmetic.
It accepts bounded odd-sized unknown chunks with their required padding. As a conservative
submission-compatibility rule, `fmt` must precede `data`; VoxReceipt rejects a technically parseable
file with the reverse order. It also rejects files above 64 MiB or 12,000,000 frames before reading
sample payloads into memory.

VoxReceipt passes an array to the adapter instead of giving the adapter an output path. This contract
lets the harness validate shape and finite values before it creates a result file. It does not
sandbox the adapter. A trusted adapter can still read files, open a network connection, or exit the
process. Review installed code before passing the trust flag.

## Pseudonymous receipts

Reports use a domain-separated `HMAC-SHA256` over the exact bytes returned by `os.fsencode(filename)`
and retain 128 digest bits. They omit the key, original filename, absolute paths, and audio samples.
One key produces the same ID for the same filename bytes across runs, which lets you compare reports
without publishing the name. NFC and NFD spellings intentionally differ. POSIX surrogate-escaped
non-UTF-8 names do not crash ID creation, although moving such names across differently encoded
filesystems may change their IDs.

An in-memory random key reduces accidental linkability. It also prevents cross-command comparison,
so the demo shares one key across its three reports. Set `VOXRECEIPT_ID_KEY` when your workflow needs
longer-lived IDs.

HMAC does not anonymize everything. Anyone with the key can test guessed filenames, and adapter name
or ledger notes can contain information you chose to provide. VoxReceipt restricts adapter metadata
to short identifiers and includes only a ledger digest summary in restoration reports. Keep the key
and full ledger under the same access controls as your experiment records.

## Content proxy boundary

The content proxy combines signed waveform correlation, frame-RMS envelope correlation, and
log-spectrum cosine similarity after a bounded delay estimate:

```text
proxy = 0.50 * waveform + 0.30 * envelope + 0.20 * spectrum
```

This mix can catch silence replacement, polarity inversion, large timing changes, and some aggressive
filtering. It can miss a replacement that shares rhythm and spectrum. It can also flag a useful
restoration that changes phase or bandwidth. A threshold therefore acts as a triage rule for your
own evaluation protocol. It does not establish semantic equivalence, speaker preservation, or
compliance with the organizer's content rule.

The proxy scores the delay-aligned overlapping samples; sample-count difference is a separate hard
gate and report field. An exact prefix can therefore receive a proxy score of 1.0 while still having
a nonzero sample delta. Exact overlapping arrays are assigned exactly 1.0 so supported numerical
backends cannot turn equality into a one-ULP pass/fail difference at the maximum threshold.

The paired evaluator reports SI-SDR change when you supply a lawful clean reference. SI-SDR also
differs from the official NISQA, SIGMOS, and DNSMOS weighted score. VoxReceipt includes none of those
models and does not estimate a leaderboard result.

## Latency boundary

VoxReceipt starts its monotonic timer immediately before `adapter.restore()` and stops it after the
method returns. The real-time factor excludes WAV decoding, PCM encoding, report creation, Python
startup, and competition-container overhead. Run repeated measurements in the target container if
you need capacity estimates.

## Data-rights boundary

The ledger forces you to record a source URI, license identifier, intended uses, human-voice flag,
competition-use decision, and evidence digest. The validator checks structure and avoids copying
entries into the restoration receipt. It cannot verify consent, license compatibility, ownership,
or jurisdiction. It opens one no-follow descriptor, requires a single-link regular file, reads at
most 1,000,000 bytes in a bounded loop, and rejects growth, replacement, oversized integers,
duplicate keys, and surrogate code points in JSON keys or string fields.

The demo avoids that ambiguity by generating harmonics, envelopes, and seeded noise. The resulting
signal contains no recorded speaker. It tests plumbing and cannot replace a speech corpus for model
training or perceptual evaluation.

## DSP baseline boundary

The `spectral-floor` adapter estimates a low-percentile power spectrum and applies bounded spectral
subtraction with overlap-add reconstruction. It gives the batch runner a nontrivial deterministic
workload. The adapter contains no learned weights and makes no quality claim. A VoiceFixer entry still
needs a participant-owned model, lawful data, and testing against the organizer's current process.

## Trade-offs

VoxReceipt rejects some WAV variants that other players accept, including `data` before `fmt`,
multiple data chunks, and bytes after the declared RIFF container. That strictness reduces parser
and submission ambiguity at the cost of format tolerance.

Report reservation, WAV creation, and report commit form one rollback boundary during `restore`.
The report target is reserved before adapter execution. Output cleanup uses the directory descriptor
held since setup and inode tokens for files created by that run, so replacing the directory pathname
does not make cleanup delete files in the replacement directory. A machine crash or `SIGKILL` cannot
run Python cleanup and can leave a reserved report or partial batch; start with an empty destination
and inspect it before retrying.

The project keeps filenames out of reports, so a failed report does not tell you which human-readable
name failed. Use the stable keyed ID inside an access-controlled mapping workflow when you need to
trace a record. Do not weaken report privacy for public artifacts.

Use the [tutorial](tutorial-getting-started.md) to inspect a safe fixture. The
[preflight guide](how-to-preflight.md) covers a local model, and the
[reference](reference.md) lists every command and field.
