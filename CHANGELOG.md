# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased] — 2.3.0

Minor release from the 2026-09-27 audit (`docs/audit/2026-09-27/PLAN.md`; each entry cites its fiche K-n). Some
public defaults change behaviour (ASK-9 = A: minor version).

### Migration notes for consumers
- **Exhaustive `switch` over the public error enums** needs a `default:` or the new cases: `VoxtralError.mlx`,
  `.unsupported`, `.missingWeights`, `.invalidTokenizer`, `.contextTooLong`; `busy` in `VoxtralPipelineError`,
  `VoxtralTTSError` and `VoxtralRealtimeError`.
- **`VoxtralPipeline.Configuration.maxTokens` is `Int?`**: code assigning an `Int` still compiles; code reading it as
  `Int` must unwrap. `nil` (the default) sizes the budget from the audio duration.
- **Calls may now throw** where they used to crash, hang or return a wrong result: `CancellationError` (cancelled
  Task), `busy` (another operation running on the same pipeline), `contextTooLong`, `missingWeights`,
  `invalidTokenizer`, `VoxtralError.mlx`.
- **Core ML encoder runs on the Neural Engine** by default (`.auto`, the default STT backend, used to hang on the GPU).
- **Dependencies**: `VoxtralCore` no longer pulls `MLXLLM`, `MLXOptimizers`, `ArgumentParser` nor the `Transformers`
  product (only `Hub`); a target relying on them transitively must declare them. `Package.resolved` is tracked.

### Changed
- **K-2 — STT memory presets keep the full context.** `MemoryOptimizationConfig` presets no longer set
  `maxKVCacheSize` (a rotating window silently dropped the start of long audio, or stopped the process beyond the
  window); the KV cache is always `KVCacheSimple`. A prompt that does not fit throws
  `VoxtralError.contextTooLong(prompt:maxTokens:limit:)` up front.
- **K-4 — STT stops on the tokenizer's special tokens** (`</s>`, `[/INST]`, read from the tokenizer), no longer on id
  32000 ("␣Capital"): transcriptions are no longer cut at the first "Capital".
- **K-5 — STT `maxTokens` is `Int?`** (was `Int`), default `nil`: the token budget follows the audio duration
  (`max(500, ⌈seconds × 6⌉ + 64)`). An explicit value stays a ceiling. Reaching it is reported, not silent:
  `lastResultTruncated` (STT), `lastTranscriptionTruncated` (Realtime). The Realtime loop runs one step per audio frame.
- **K-6 — `downloadModel(modelId:revision:)`**, a placeholder, now throws `VoxtralError.unsupported` instead of
  pretending to download (use `ModelDownloader`).
- **K-10 — `VoxtralPipeline.Model.repoId` comes from `ModelRegistry`**: `small-24b-8bit` is
  `VincentGOURBIN/voxtral-small-8bit` (was `mzbac/Voxtral-Small-24B-2507-8bit` in the enum, ASK-15), and `loadModel()`
  resolves by id, so a downloaded model loads offline without a Hub request.
- **K-9 — `realtime-4b` (original Mistral checkpoint) loads**: it downloads only `consolidated.safetensors`,
  `params.json` and `tekken.json` (8.87 GB instead of 17.72), the transformers `config.json` is skipped for
  `params.json`, and `VoxtralRealtimePipeline.loadModel(modelId:)` throws on an unknown id instead of loading the default.
- **K-11 — one operation at a time per pipeline**: a second call while a transcription, a synthesis or a voice
  enrollment runs is refused with `busy` instead of racing (the enrollment's gradient with an inference could
  deadlock the process). Voice enrollment no longer draws from the global RNG shared with synthesis.
- **K-12 — `synthesizeStreaming` is really progressive**: the stream is returned at once and produced in a Task;
  stopping the consumer (break, cancelled Task) stops the generation within one frame and the pipeline returns to
  `.ready`. Before, the whole generation ran before the first chunk.
- **K-14 — TTS frame cap proportional to the text**: `Configuration.framesPerTextToken` (10.4) and `framesCapBase`
  (70) make the effective cap `min(maxFrames, 70 + ⌈10.4 × text tokens⌉)` (batch and streaming), so a missed end of
  audio stops at about 3 × the expected duration instead of 2 500 frames (200 s); `framesPerTextToken = nil` restores
  the fixed cap. `lastSynthesisTruncated` tells when the cap was reached.
- **K-15 — work off Swift's cooperative pool**: model loading and generation run on a dedicated queue; a cancelled
  Task stops a transcription, a synthesis or a Realtime run within one step (< 200 ms measured on 11 min of audio)
  with `CancellationError`, the pipeline back to `.ready`.
- **K-24 — STT and Realtime downloads skip `consolidated.safetensors`** (a second copy of the weights their
  loaders never read: Mini 3B 9.36 GB instead of 18.7 GB, Small 24B 48.5 instead of 97 GB); registry `size` shows the
  exact size and `quantization` the real precision (`bfloat16` for the Mistral STT packs); the Core ML encoder variant
  follows the model's `config.json` when the id is a local folder.
- **K-23 — the library is quiet**: nothing on stdout unless `VoxtralDebug.enabled` (messages also go to the unified
  log, subsystem `com.vincentgourbin.voxtral`); `writeDebugToDump` no longer appends to
  `/tmp/swift_debug_generation.txt`; no model or Core ML lookup in the current directory anymore.
- **K-25 — Core ML encoder** downloaded under `ModelDownloader.customModelsDirectory`, reloadable offline, with
  explicit errors; the hybrid encoder no longer loads Core ML when `.mlx` is requested and refuses to encode with an
  unloaded MLX encoder.
- **K-26 — voice enrollment is reproducible**: the same `Config.seed` and reference give the same codes (the
  spectral loss had a non-deterministic GPU gradient). Without a seed, runs still differ.
- **K-32 — `fullCleanup()` no longer resets the MLX peak-memory counter** (the field `resetPeakMemory` stays and has
  no effect): measuring tools reset it themselves.
- **K-32b — `VoxtralCoreMLConfig` presets `default`, `mini` and `small` use `.cpuAndNeuralEngine`**: on the GPU
  (MPSGraph) Core ML deadlocked with MLX in the same process, the first prediction waiting forever for a Metal command
  buffer (backend `.auto`, the default, hung). `gpuOnly` stays on the GPU and must not share a process with MLX.

### Added
- **K-1 — `VoxtralError.mlx(String)`**: an MLX error raised inside a public entry point (transcribe, chat,
  synthesis, Realtime, loading) is thrown instead of terminating the host process.
- **K-2 — `VoxtralError.contextTooLong(prompt:maxTokens:limit:)`.**
- **K-4 — `VoxtralForConditionalGeneration.stopTokenIds`**, set by the pipeline from its tokenizer.
- **K-5 — `lastResultTruncated`** (STT) and **`lastTranscriptionTruncated`** (Realtime).
- **K-6 — `VoxtralError.unsupported(String)`** and a download manifest (`.voxtral-complete.json`, SHA-256 per file):
  a model folder counts as downloaded only when its manifest is complete.
- **K-7 — `VoxtralError.missingWeights([String])`** (weights checked against the model's keys and shapes),
  **`VoxtralError.invalidTokenizer(String)`** and **`TekkenTokenizer.load(modelPath:) throws`**.
- **K-11 — `busy(String)`** in `VoxtralPipelineError`, `VoxtralTTSError` and `VoxtralRealtimeError`.
- **K-14 — `VoxtralTTSPipeline.frameCap(forText:)`, `textTokenCount(_:)`, `lastSynthesisTruncated`.**
- **K-16 — `VoxtralMemoryManager.optimizeIfNeeded(tokenIndex:config:)`**: the pipeline passes its own memory
  configuration instead of writing the shared one.
- **K-26 — `VoxtralVoiceEnrollment.Config.seed`, `checkpointURL`, `checkpointEvery`** and
  `EnrollmentCheckpointError`: an interrupted enrollment resumes from its checkpoint and ends with the codes of an
  uninterrupted run. CLI `enroll`: `--seed`, `--checkpoint`, `--checkpoint-every`, `--stop-after`.
- **K-32 — `VoxtralCLI bench`** (`stt`, `tts`, `realtime`, `enroll`, `chat`): one JSON line per measured pass, A/A
  dispersion; `--trace` exports a Chrome trace of a separate diagnostic pass (K-32b).
- **K-32b — `VoxtralRealtimePipeline.lastPadFraction`**: share of the last transcription's decode steps that carry
  no text (control tokens such as `[STREAMING_PAD]`).
- **K-52 — opt-in MLX cache limit**: `cacheLimitBytes: Int?` in `MemoryOptimizationConfig` and in the TTS and Realtime
  configurations (`nil` by default: the host's MLX setting is untouched). When set, the limit applies after loading
  and the host's value is restored at `unload()`.

- **K-23 — `VoxtralDebug.console(_:)`** for output a caller asked for (model listings).
- **K-27 — `VoxtralPipeline.lastTokenCount`**, `loadVoxtralStandardModel(modelPath:)`, `EnrollmentLossComputer(validating:)`.
- **K-24 — registries**: `approximateBytes` (exact bytes of the downloaded weights) on `VoxtralModelInfo`,
  `VoxtralTTSModelInfo` and `VoxtralRealtimeModelInfo`; `ModelDownloader.downloadRepoDirect(…, excluding:)` and
  `downloadByRepoId(_:excluding:progress:)`; `VoxtralCoreMLVariant.variant(forConfigAt:)`.

- **K-9 — `VoxtralRealtimeModelInfo.files`**: the exact repository files an entry downloads.

### Deprecated
- **K-30 — legacy Python-port family and dead public code** (ASK-23: deprecated in 2.3, removed in 3.0; each message
  names the replacement): `VoxtralGenerator` (stops the process at load) and its extensions,
  `VoxtralGenerationParameters`, the library `VoxtralCLI` class, `loadVoxtralModel(modelPath:dtype:lazy:)`,
  `loadVoxtralModel(modelPath:dtype:)`, `loadVoxtralModelWithMLXLM`, `downloadModel(modelId:revision:)`,
  `loadConfig(modelPath:)`, `loadWeights(modelPath:)`, `VoxtralForConditionalGeneration.init(path:)`,
  `init(config: PythonVoxtralConfig)`, `fromPretrained(_:)`, `customLoadWeights`, `replaceAllQuantizedLinearWithWeights`,
  `VoxtralMultiModalProjector.replaceQuantizedLinearWithWeights`, `LlamaModelWrapper`, `mlxLMCreateAttentionMask`,
  `mlxLMScaledDotProductAttention` (×2), `mlxLMInitializeRope`, `mlxLMGetModelPath`, `createCausalMask(N:…)`,
  `quantizeModel`, `saveConfig`, `saveModel`, `treeReduce`, `treeFlatten`, `computeBitsPerWeight` (×2),
  `voxtralMixedQuantizationPredicate` (×2), `loadQuantizedVoxtral` (the `VoxtralQuantization` overload),
  `getQuantizationStats`, `AudioEncoder`, `ChatTemplateProcessor`, `TekkenTokenizer.encodeTranscription`.
  `language_model` no longer accepts a `LlamaModelWrapper` (never built by the loaders).
- **K-27** — `VoxtralTranscriptionManager.chat(systemPrompt:userMessage:)` (always throws `audioRequired`),
  `saveQuantizedModel` (writes only `config.json`), `MLXCoreMLBridge.toMLMultiArrayNoCopy` (copies),
  `VoxtralForConditionalGeneration.init(officialLlama:config:)` (legacy decoder, random `lm_head`),
  `ModelDownloader.hubApi` / `reconfigureHubApi()` (downloads no longer use HubApi),
  `loadVoxtralStandardModel(modelPath:dtype:)` (`dtype` ignored; use `loadVoxtralStandardModel(modelPath:)`),
  `EnrollmentLossComputer(reference:)` (stops on a short reference; use `init(validating:)`).
- **K-7 — `TekkenTokenizer(modelPath:)`**: falls back silently to a demo vocabulary; use `TekkenTokenizer.load(modelPath:)`.
- **K-26 — `VoxtralVoiceEnrollment.optimize(reference:progress:)`** (non-throwing): ignores divergence, cancellation
  and checkpoints; use `optimize(reference:progress:shouldContinue:)`.

### Removed
- **K-32 — the `VoxtralBenchmark` executable** (ASK-26 = B), replaced by `VoxtralCLI bench`.

### Fixed
- **K-27 — no crash from the public API**: `TranscriptionResult.tokenCount` counts the generated tokens (was 0); no
  forced casts; an unsupported module type, a missing input or a too short reference throws
  `invalidConfiguration` at the entry points instead of stopping the process; unreadable tokenizer or quantization
  configs throw instead of being ignored.
- **K-3 — attention masks built by the cache** (boolean, shaped like the keys): a bf16 STT model no longer fails
  with "Mask type must promote to output type", and the legacy decoder no longer stops beyond its first 512-position
  prefill chunk. Outputs unchanged (logits identical).
- **K-13 — Realtime beyond 15 s**: the encoder attends within its 750-position sliding window (chunks with a rotating
  KV cache) and the decoder keeps its 8 192-step window; output no longer degenerates after about 30 s.
- **K-13 — `TekkenTokenizer.decode(skipSpecialTokens: true)` skips every control token** (ids below the special-token
  count), not only BOS/EOS/PAD: Realtime output no longer contains NUL bytes between words.
- **K-16 — shared state under locks** (mel filter cache, debug flags, `customModelsDirectory`, memory manager
  configuration) and results evaluated before they cross actors (`TTSSynthesisResult`, `TTSStreamingChunk`,
  `GenerationChunk`): no more data races between pipelines.

### Dependencies
- **K-22** — `VoxtralCore` depends on mlx-swift, mlx-swift-lm (`MLXLMCommon`), `Hub` and swift-mlx-profiler only;
  swift-mlx-profiler `from: "1.5.1"`; `Package.resolved` tracked (mlx-swift 0.31.6 `0bb916c`, mlx-swift-lm
  `main@604fae7`, swift-mlx-profiler 1.5.1 `bfe71d8`); 28 `@available` below the platform floor removed.
