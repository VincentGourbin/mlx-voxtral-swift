# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased] — 2.3.0

Minor release from the 2026-09-27 audit (`docs/audit/2026-09-27/PLAN.md`). Some public defaults change behaviour
(ASK-9 = A: minor version, FluxForge Studio notified). New cases in public error enums: an exhaustive `switch`
over these enums needs a `default:` or the new cases.

### Changed
- **K-2 — STT memory presets keep the full context.** `MemoryOptimizationConfig` presets no longer set
  `maxKVCacheSize` (a rotating window silently dropped the start of long audio, or stopped the process beyond the
  window); the KV cache is always `KVCacheSimple`. A prompt that does not fit throws
  `VoxtralError.contextTooLong(prompt:maxTokens:limit:)` instead.
- **K-5 — STT `maxTokens` is `Int?`** (was `Int`), default `nil`: the token budget follows the audio duration
  (`max(500, ⌈seconds × 6⌉ + 64)`). An explicit value stays a ceiling. Reaching it is reported, not silent:
  `lastResultTruncated` (STT), `lastTranscriptionTruncated` (Realtime). The Realtime loop runs one step per audio frame.

### Added
- **K-1 — `VoxtralError.mlx(String)`**: an MLX error raised inside a public entry point (transcribe, chat,
  synthesis, Realtime) is thrown instead of terminating the host process.
- **K-6 — `VoxtralError.unsupported(String)`** and a download manifest (`.voxtral-complete.json`, SHA-256 per file):
  a model folder counts as downloaded only when its manifest is complete. The placeholder `downloadModel(modelId:)`
  throws `.unsupported`.
- **K-7 — `VoxtralError.missingWeights([String])`** (weights checked against the model's keys and shapes) and
  **`VoxtralError.invalidTokenizer(String)`**: `TekkenTokenizer.load(modelPath:)` throws instead of falling back to a
  demo vocabulary.
- **K-11 — `busy(String)`** in `VoxtralPipelineError`, `VoxtralTTSError` and `VoxtralRealtimeError`: a second call
  while an operation (or voice enrollment) runs is refused instead of racing.
- **K-52 — opt-in MLX cache limit**: `cacheLimitBytes: Int?` in `MemoryOptimizationConfig` and in the TTS and Realtime
  configurations (`nil` by default: the host's MLX setting is untouched). When set, the limit applies after loading
  and the host's value is restored at `unload()`.

### Fixed
- **K-13 — Realtime beyond 15 s**: the encoder attends within its 750-position sliding window (chunks with a rotating
  KV cache) and the decoder keeps its 8 192-step window; output no longer degenerates after about 30 s.
- **K-13 — `TekkenTokenizer.decode(skipSpecialTokens: true)` skips every control token** (ids below the special-token
  count), not only BOS/EOS/PAD: Realtime output no longer contains NUL bytes between words.
