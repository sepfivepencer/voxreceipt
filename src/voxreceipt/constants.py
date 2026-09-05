"""Competition constraints and VoxReceipt report schema identifiers."""

SAMPLE_RATE = 48_000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
MAX_SAMPLE_DELTA = 3_000
PCM_COMPRESSION = "NONE"

# Resource ceilings are intentionally lower than RIFF's 32-bit theoretical limit. They keep
# malformed or sparse containers from turning a header inspection into a multi-gigabyte read.
MAX_WAV_BYTES = 64 * 1024 * 1024
MAX_WAV_FRAMES = 12_000_000
MAX_LEDGER_BYTES = 1_000_000

PREFLIGHT_SCHEMA = "voxreceipt.preflight.v1"
RESTORE_SCHEMA = "voxreceipt.restore.v1"
EVALUATION_SCHEMA = "voxreceipt.evaluation.v1"
LEDGER_SCHEMA = "voxreceipt.ledger.v1"

OFFICIAL_RULES_URL = "https://challenge.xfyun.cn/2020/ai-contest/api/page-data/VoiceFixer"
RULES_CHECKED_DATE = "2026-09-05"
