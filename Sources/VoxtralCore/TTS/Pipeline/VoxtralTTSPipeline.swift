/**
 * VoxtralTTSPipeline - Simplified facade API for Voxtral TTS
 *
 * Usage:
 * ```swift
 * let pipeline = VoxtralTTSPipeline()
 * try await pipeline.loadModel()
 * let result = try await pipeline.synthesize(text: "Hello!", voice: .neutralFemale)
 * try WAVWriter.write(waveform: result.waveform, to: outputURL)
 * pipeline.unload()
 * ```
 */

import Foundation
import MLX
import MLXLMCommon
import MLXProfiler

public class VoxtralTTSPipeline: @unchecked Sendable {

    // MARK: - Configuration

    public struct Configuration: Sendable {
        public var maxFrames: Int
        public var temperature: Float
        public var cfgAlpha: Float
        public var flowSteps: Int
        /// Sanitize text before synthesis (lowercase ALL-CAPS, add terminal punctuation, etc.)
        /// Disable if you need precise control over intonation via casing/punctuation.
        public var sanitizeText: Bool
        /// Trim low-energy lead-in silence frames from the beginning of generated audio.
        /// Applies to `synthesize`/`synthesizeToFile` only — `synthesizeStreaming`
        /// yields chunks as they decode and never trims.
        public var trimLeadIn: Bool
        /// Trim low-energy trailing silence frames from the end of generated
        /// audio (opt-in; useful when downstream consumers align on speech
        /// boundaries, e.g. lip-sync video generation). Like `trimLeadIn`,
        /// ignored by `synthesizeStreaming`.
        public var trimTail: Bool
        /// MLX buffer-cache limit set after loading and restored at `unload()`; nil leaves the host's
        /// process-wide setting alone (K-52)
        public var cacheLimitBytes: Int?
        /// Frames allowed per text token on top of `framesCapBase`: the effective cap is
        /// min(maxFrames, framesCapBase + ⌈framesPerTextToken × text tokens⌉), so a missed end of audio costs about
        /// 3 × the expected duration instead of `maxFrames` (K-14). nil keeps the fixed `maxFrames`.
        public var framesPerTextToken: Float?
        public var framesCapBase: Int

        /// 3 × the fit frames = 23.5 + 3.477 × text tokens over 12 texts × 3 seeds × 3 packs (4 / 6 / bf16, K-14),
        /// rounded down: every normal synthesis stays under the cap, the one missed end of audio (517 frames for
        /// 7 tokens) stops at 143
        public static let defaultFramesPerTextToken: Float = 10.4
        public static let defaultFramesCapBase = 70

        public static var `default`: Configuration {
            Configuration(maxFrames: 2500, temperature: 0.0, cfgAlpha: 1.2, flowSteps: 8, sanitizeText: true, trimLeadIn: true, trimTail: false)
        }

        public init(maxFrames: Int = 2500, temperature: Float = 0.0, cfgAlpha: Float = 1.2, flowSteps: Int = 8, sanitizeText: Bool = true, trimLeadIn: Bool = true, trimTail: Bool = false, cacheLimitBytes: Int? = nil,
                    framesPerTextToken: Float? = Self.defaultFramesPerTextToken,
                    framesCapBase: Int = Self.defaultFramesCapBase) {
            self.cacheLimitBytes = cacheLimitBytes
            self.framesPerTextToken = framesPerTextToken
            self.framesCapBase = framesCapBase
            self.maxFrames = maxFrames
            self.temperature = temperature
            self.cfgAlpha = cfgAlpha
            self.flowSteps = flowSteps
            self.sanitizeText = sanitizeText
            self.trimLeadIn = trimLeadIn
            self.trimTail = trimTail
        }
    }

    // MARK: - State

    public enum State: Sendable {
        case unloaded, loading, ready, synthesizing, error(String)

        var isUnloaded: Bool { if case .unloaded = self { return true }; return false }
        var isReady: Bool { if case .ready = self { return true }; return false }
    }

    // MARK: - Properties

    /// Recommended `warmUpText` for enrolled voices (A6b): a short vocalise that
    /// both covers the first-sentence degradation and stabilises the whole
    /// generation across seeds. Picked by blind A/B over vocalise/verbal/hum
    /// carriers; pair with `warmUpLeadInFrames: 0`.
    public static let recommendedWarmUpVocalise = "La la la la la la la la."

    public var configuration: Configuration
    /// Read-only view of the gate: every transition happens under its lock (K-11)
    public var state: State { gate.state }

    /// True when the last synthesis reached its frame cap without an end of audio (K-14)
    public private(set) var lastSynthesisTruncated = false

    /// Text tokens the model reads for `text` (after sanitization when enabled)
    public func textTokenCount(_ text: String) -> Int? {
        guard let tokenizer else { return nil }
        let processed = configuration.sanitizeText ? VoxtralTTSModel.sanitizeTextForTTS(text) : text
        return tokenizer.encode(processed).count
    }

    /// Frame cap applied to `text`: min(maxFrames, framesCapBase + ⌈framesPerTextToken × text tokens⌉) (K-14)
    public func frameCap(forText text: String) -> Int {
        guard let tokens = textTokenCount(text) else { return configuration.maxFrames }
        return Self.frameCap(textTokens: tokens, configuration: configuration)
    }

    static func frameCap(textTokens: Int, configuration: Configuration) -> Int {
        guard let perToken = configuration.framesPerTextToken else { return configuration.maxFrames }
        return min(configuration.maxFrames, configuration.framesCapBase + Int((perToken * Float(textTokens)).rounded(.up)))
    }
    public let sampleRate: Int = 24000

    /// State, running operation and generation token, changed atomically
    let gate = PipelineGate<State>(.unloaded)

    /// Opt-in MLX cache limit held while loaded (K-52)
    private let cachePolicy = MLXCachePolicy()

    private(set) var ttsModel: VoxtralTTSModel?
    private(set) var tokenizer: TekkenTokenizer?
    private let voiceManager: VoxtralVoicePresetManager
    private var modelDirectory: URL?
    private(set) var voiceEmbeddings: [String: MLXArray] = [:]
    /// Registry id of the loaded model (for the activity beacon manifests).
    private var loadedModelID: String?

    // Cached voice-conditioned prefill KV for the last-used voice. The voice
    // frames precede the text in the prompt, so their KV depends only on the
    // voice — reusing it skips the voice prefill on repeated syntheses.
    // Invariant: this entry is never mutated by generation — generate() and
    // generateStreaming() clone it (cloneKVCaches) before prefilling, so
    // consecutive syntheses always start from the pristine voice prefix
    // (guarded by KVCacheCloneTests + TTSConsecutiveSynthesisReproTests).
    private var prefixCacheEntry: (key: String, cache: [any KVCache], len: Int)?

    /// Get-or-compute the voice prefix KV cache for `key`.
    private func voicePrefix(_ model: VoxtralTTSModel, for voiceEmb: MLXArray, key: String)
        -> (cache: [any KVCache], len: Int)
    {
        if let e = prefixCacheEntry, e.key == key { return (e.cache, e.len) }
        let (cache, len) = model.precomputeVoicePrefixCache(voiceEmbedding: voiceEmb)
        prefixCacheEntry = (key, cache, len)
        return (cache, len)
    }

    public typealias ProgressCallback = @Sendable (Double, String) -> Void

    /// Post-process one decoded waveform per the configuration's trim flags.
    private func applyTrims(_ raw: MLXArray) -> MLXArray {
        var waveform = configuration.trimLeadIn ? trimLeadInSilence(raw, sampleRate: sampleRate) : raw
        if configuration.trimTail {
            waveform = trimTrailingSilence(waveform, sampleRate: sampleRate)
        }
        return waveform
    }

    // MARK: - Initialization

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
        self.voiceManager = VoxtralVoicePresetManager()
    }

    // MARK: - Model Loading

    public func loadModel(modelInfo: VoxtralTTSModelInfo? = nil, progress: ProgressCallback? = nil) async throws {
        // MLX errors become VoxtralError.mlx instead of terminating the host (K-1)
        try await withMLXErrors { _ in
            // Atomic check-and-set: a concurrent load or a running operation is refused (K-11)
            let generation = try gate.begin(
                "loading", accepts: { current in current.isUnloaded || { if case .error = current { return true }; return false }() },
                refusal: VoxtralTTSError.invalidConfiguration("Model already loaded or loading"),
                busy: VoxtralTTSError.busy, state: .loading, newGeneration: true)
            prefixCacheEntry = nil  // a new model invalidates any cached voice prefix

            let resolvedInfo = modelInfo ?? VoxtralTTSRegistry.defaultModel
            let beacon = RuntimeBeacon.begin(task: "load-tts-model", model: resolvedInfo.id)
            defer { beacon?.end() }

            do {
                let session = MLXProfiler.shared.activeSession

                progress?(0.05, "Resolving TTS model...")
                session?.beginPhase("1. Model Download", category: .modelLoad)
                let modelInfo = resolvedInfo
                let modelDir = try await ModelDownloader.downloadTTSModel(modelInfo) { p, msg in
                    progress?(0.05 + p * 0.35, msg)
                }
                self.modelDirectory = modelDir
                session?.endPhase("1. Model Download", category: .modelLoad)

                // Tokenizer first: a missing or invalid tekken.json fails before gigabytes of weights (K-7)
                progress?(0.35, "Loading tokenizer...")
                session?.beginPhase("3. Tokenizer Loading", category: .tokenization)
                // TekkenTokenizer expects the MODEL DIRECTORY, not the tekken.json file path
                self.tokenizer = try TekkenTokenizer.load(modelPath: modelDir.path)
                session?.endPhase("3. Tokenizer Loading", category: .tokenization)

                progress?(0.40, "Loading TTS model...")
                session?.beginPhase("2. Model Loading", category: .modelLoad)
                // Weights load off the cooperative pool (K-15); MLX errors caught on that queue (K-1)
                let model = try await runOffCooperativePool {
                    try withMLXErrors { _ in
                        try loadVoxtralTTSModel(from: modelDir) { p, msg in progress?(0.40 + Double(p) * 0.40, msg) }
                    }
                }
                self.ttsModel = model
                session?.endPhase("2. Model Loading", category: .modelLoad)

                // Load voice embeddings
                progress?(0.90, "Loading voice embeddings...")
                session?.beginPhase("4. Voice Embeddings", category: .voiceEmbedding)
                let voiceDir = modelDir.appendingPathComponent("voice_embedding")
                for voice in VoxtralVoice.allCases {
                    let safetensorsPath = voiceDir.appendingPathComponent("\(voice.rawValue).safetensors")
                    if FileManager.default.fileExists(atPath: safetensorsPath.path) {
                        let data = try MLX.loadArrays(url: safetensorsPath)
                        if let emb = data["embedding"] ?? data.values.first {
                            voiceEmbeddings[voice.rawValue] = emb
                        }
                    }
                }
                session?.endPhase("4. Voice Embeddings", category: .voiceEmbedding)

                progress?(1.0, "TTS model ready (\(voiceEmbeddings.count) voices loaded)")
                loadedModelID = resolvedInfo.id
                cachePolicy.apply(configuration.cacheLimitBytes)
                gate.end(generation, state: .ready)

            } catch {
                gate.end(generation, state: .error(error.localizedDescription))
                throw error
            }
        }
    }

    // MARK: - Synthesis

    public func synthesize(
        text: String,
        voice: VoxtralVoice = .neutralFemale,
        seed: UInt64? = nil
    ) async throws -> TTSSynthesisResult {
        // MLX errors become VoxtralError.mlx instead of terminating the host (K-1)
        // Off the cooperative pool, cancellable at every step (K-15)
        return try await runOffCooperativePool { [self] in try withMLXErrors { _ in
            let generation = try gate.begin(
                "synthesis", accepts: { $0.isReady }, refusal: VoxtralTTSError.invalidConfiguration("Model not loaded"),
                busy: VoxtralTTSError.busy, state: .synthesizing)
            defer {
                cachePolicy.endOfResponse()
                gate.end(generation, state: .ready)
            }
            guard let model = ttsModel, let tokenizer else {
                throw VoxtralTTSError.invalidConfiguration("Model not loaded")
            }
            try model.validateModuleTypes()  // unsupported module types throw here (K-27)

            guard let voiceEmb = voiceEmbeddings[voice.rawValue] else {
                throw VoxtralTTSError.voiceNotFound("Voice '\(voice.rawValue)' not loaded")
            }

            let startTime = Date()
            let profiler = MLXProfiler.shared
            let session = profiler.activeSession
            let beacon = RuntimeBeacon.begin(task: "tts", model: loadedModelID)
            defer { beacon?.end() }

            let prefix = voicePrefix(model, for: voiceEmb, key: voice.rawValue)
            do {
                // Generate audio codes (semantic code generation + flow matching inside)
                profiler.startSemanticGen()
                let cap = frameCap(forText: text)
                let (codes, numFrames, ttft) = model.generate(
                    text: text,
                    voiceEmbedding: voiceEmb,
                    tokenizer: tokenizer,
                    maxTokens: cap,
                    sanitize: configuration.sanitizeText,
                    seed: seed,
                    prefixCache: prefix.cache,
                    prefixLen: prefix.len
                )
                profiler.endSemanticGen(frameCount: numFrames)
                try VoxtralCancellation.check()  // generation stopped early for a cancelled caller (K-15)
                lastSynthesisTruncated = numFrames >= cap
                profiler.setTTFT(ttft)

                guard numFrames > 0 else {
                    throw VoxtralTTSError.synthesisError("No audio frames generated")
                }

                // Decode to waveform, optionally trim lead-in silence
                profiler.startCodecDecode()
                let rawWaveform = model.decodeToWaveform(codes)
                MLX.eval(rawWaveform)
                profiler.endCodecDecode()

                session?.beginPhase("Audio Post-processing", category: .postProcess)
                let waveform = applyTrims(rawWaveform)
                session?.endPhase("Audio Post-processing", category: .postProcess)

                let generationTime = Date().timeIntervalSince(startTime)
                let audioDuration = Double(waveform.dim(0)) / Double(sampleRate)
                profiler.setAudioDuration(audioDuration)

                return TTSSynthesisResult(
                    waveform: waveform,
                    numFrames: numFrames,
                    sampleRate: sampleRate,
                    generationTime: generationTime,
                    timeToFirstToken: ttft
                )
            } catch {
                throw error
            }
        }
        }
    }

    @discardableResult
    public func synthesizeToFile(
        text: String,
        voice: VoxtralVoice = .neutralFemale,
        outputURL: URL
    ) async throws -> TTSSynthesisResult {
        let result = try await synthesize(text: text, voice: voice)
        try WAVWriter.write(waveform: result.waveform, to: outputURL, sampleRate: sampleRate)
        return result
    }

    // MARK: - ZeroVoice Synthesis

    /// Synthesize speech using a ZeroVoice coordinate (procedural voice).
    public func synthesize(
        text: String,
        voiceCoordinate: (x: Int, y: Int, z: Int)
    ) async throws -> TTSSynthesisResult {
        guard let voiceEmb = zeroVoice?.voiceAt(x: voiceCoordinate.x, y: voiceCoordinate.y, z: voiceCoordinate.z) else {
            throw VoxtralTTSError.voiceNotFound("Could not generate voice for coordinate (\(voiceCoordinate.x), \(voiceCoordinate.y), \(voiceCoordinate.z))")
        }
        return try await synthesize(text: text, voiceEmbedding: voiceEmb)
    }

    /// Synthesize speech using a pre-computed blended voice embedding.
    ///
    /// Pass `seed` for reproducible output: the acoustic flow-matching step
    /// samples random noise, so without a seed the same text+voice yields a
    /// different waveform (and a different length) on every call.
    ///
    /// Pass `warmUpText` (A6b mitigation) to prepend a short throwaway utterance
    /// that absorbs the enrolled-voice first-sentence degradation; its audio is
    /// trimmed off (see `trimLeadingCarrier`) so the returned waveform starts on
    /// the real `text`. A short **vocalise** works best — pass
    /// `recommendedWarmUpVocalise` (`"La la la la la la la la."`); a uniform
    /// sound both covers the warm-up AND stabilises the whole generation across
    /// seeds. A verbal carrier is less consistent and a hum/"ah-ah" scored
    /// slightly worse in blind tests. Keep it single-clause; avoid many "…"
    /// which makes the model over-generate.
    /// `warmUpLeadInFrames` keeps that many 80 ms frames of the carrier's
    /// terminal silence before the content — 0 (tight cut) is the recommended
    /// default (blind-test winner); 3 (~0.24 s) adds a small breath.
    public func synthesize(
        text: String,
        voiceEmbedding: MLXArray,
        seed: UInt64? = nil,
        warmUpText: String? = nil,
        warmUpLeadInFrames: Int = 0
    ) async throws -> TTSSynthesisResult {
        // MLX errors become VoxtralError.mlx instead of terminating the host (K-1)
        // Off the cooperative pool, cancellable at every step (K-15)
        return try await runOffCooperativePool { [self] in try withMLXErrors { _ in
            let generation = try gate.begin(
                "synthesis", accepts: { $0.isReady }, refusal: VoxtralTTSError.invalidConfiguration("Model not loaded"),
                busy: VoxtralTTSError.busy, state: .synthesizing)
            defer {
                cachePolicy.endOfResponse()
                gate.end(generation, state: .ready)
            }
            guard let model = ttsModel, let tokenizer else {
                throw VoxtralTTSError.invalidConfiguration("Model not loaded")
            }
            try model.validateModuleTypes()  // unsupported module types throw here (K-27)

            let startTime = Date()
            let profiler = MLXProfiler.shared
            let session = profiler.activeSession
            let beacon = RuntimeBeacon.begin(task: "tts", model: loadedModelID)
            defer { beacon?.end() }

            // Prepend the warm-up carrier as its own sentence so the model puts a
            // detectable pause between it and the real content.
            let genText: String
            if let warmUpText, !warmUpText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let carrier = warmUpText.trimmingCharacters(in: .whitespacesAndNewlines)
                let sep = carrier.last.map { ".!?…".contains($0) } == true ? " " : ". "
                genText = carrier + sep + text
            } else {
                genText = text
            }

            do {
                profiler.startSemanticGen()
                let cap = frameCap(forText: genText)
                let (codes, numFrames, ttft) = model.generate(
                    text: genText,
                    voiceEmbedding: voiceEmbedding,
                    tokenizer: tokenizer,
                    maxTokens: cap,
                    sanitize: configuration.sanitizeText,
                    seed: seed
                )
                profiler.endSemanticGen(frameCount: numFrames)
                try VoxtralCancellation.check()  // generation stopped early for a cancelled caller (K-15)
                lastSynthesisTruncated = numFrames >= cap
                profiler.setTTFT(ttft)

                guard numFrames > 0 else {
                    throw VoxtralTTSError.synthesisError("No audio frames generated")
                }

                profiler.startCodecDecode()
                let rawWaveform = model.decodeToWaveform(codes)
                MLX.eval(rawWaveform)
                profiler.endCodecDecode()

                session?.beginPhase("Audio Post-processing", category: .postProcess)
                // Drop the warm-up carrier's audio. When it trims, the carrier trim
                // already positions the content start (including any kept
                // `warmUpLeadInFrames` breath), so DON'T also run trimLeadInSilence —
                // it would strip that lead-in back off. Apply only the tail trim.
                // Locate the cut with a purely ABSOLUTE silence floor rather than
                // the default peak-relative threshold. An enrolled voice renders
                // the carrier much quieter than the content, so a peak-relative
                // threshold lands ABOVE the carrier: the "skip leading silence"
                // scan then consumes the carrier *and* its terminal pause, and the
                // first gap it finds is the pause after the first sentence — which
                // is cut away with the carrier (measured: a 2-sentence text lost
                // its whole first sentence, 18.9 s → 11.8 s). The carrier's
                // terminal pause is true digital silence (~−110 dB), far below any
                // speech, so a fixed low floor isolates it whatever the content
                // loudness.
                let (carrierTrimmed, carrierCut) = genText != text
                    ? trimLeadingCarrierAdaptive(rawWaveform, sampleRate: sampleRate,
                                                 leadInFrames: warmUpLeadInFrames)
                    : (rawWaveform, 0)
                let waveform: MLXArray
                if carrierCut > 0 {
                    waveform = configuration.trimTail
                        ? trimTrailingSilence(carrierTrimmed, sampleRate: sampleRate)
                        : carrierTrimmed
                } else {
                    waveform = applyTrims(rawWaveform)
                }
                session?.endPhase("Audio Post-processing", category: .postProcess)

                let generationTime = Date().timeIntervalSince(startTime)
                let audioDuration = Double(waveform.dim(0)) / Double(sampleRate)
                profiler.setAudioDuration(audioDuration)

                return TTSSynthesisResult(
                    waveform: waveform,
                    numFrames: numFrames,
                    sampleRate: sampleRate,
                    generationTime: generationTime,
                    timeToFirstToken: ttft
                )
            } catch {
                throw error
            }
        }
        }
    }

    /// ZeroVoice generator (lazy-initialized from loaded voice embeddings).
    public var zeroVoice: VoxtralZeroVoice? {
        guard !voiceEmbeddings.isEmpty else { return nil }
        return VoxtralZeroVoice(voiceEmbeddings: voiceEmbeddings)
    }

    /// Get the recipe for a ZeroVoice coordinate (metadata, no computation).
    public func voiceRecipe(x: Int, y: Int, z: Int) -> VoiceRecipe? {
        zeroVoice?.voiceRecipe(x: x, y: y, z: z)
    }

    // MARK: - Voice Enrollment (cloning)

    /// Clone a voice from a reference recording and write a voice embedding
    /// `.safetensors` (key "embedding", shape [T+1, 3072]) usable with
    /// `synthesize(text:voiceEmbedding:)` or `--voice-embedding`.
    /// Offline: ~30 min for 5000 epochs on an M-series Mac.
    ///
    /// `shouldContinue` is polled every epoch; return `false` to cancel the
    /// run — the optimization stops within one epoch and `CancellationError`
    /// propagates from inside the loop, so no partial embedding file is ever
    /// written (the cancellation decision is the in-loop poll itself, not a
    /// second read of the predicate afterwards).
    @discardableResult
    public func enrollVoice(
        referenceURL: URL,
        outputURL: URL,
        config: VoxtralVoiceEnrollment.Config = .init(),
        progress: ((VoxtralVoiceEnrollment.Progress) -> Void)? = nil,
        shouldContinue: (() -> Bool)? = nil
    ) throws -> MLXArray {
        // MLX errors become VoxtralError.mlx instead of terminating the host (K-1)
        return try withMLXErrors { _ in
            // The enrollment (gradient) holds the pipeline: a synthesis or a load meanwhile gets
            // `busy` instead of racing it into the compile × vjp deadlock (K-11, piège 20)
            let generation = try gate.begin(
                "enrollment", accepts: { $0.isReady }, refusal: VoxtralTTSError.invalidConfiguration("Model not loaded"),
                busy: VoxtralTTSError.busy)
            defer { gate.end(generation) }
            guard let model = ttsModel else {
                throw VoxtralTTSError.invalidConfiguration("Model not loaded")
            }
            let enroller = VoxtralVoiceEnrollment(model: model, config: config)
            let reference = try enroller.prepareReference(url: referenceURL)
            // Always go through the throwing overload (a nil `shouldContinue`
            // becomes a never-cancel poll) so a diverged run surfaces as an error
            // on every path, GUI included, instead of silently saving a bad voice.
            let codes = try enroller.optimize(
                reference: reference, progress: progress, shouldContinue: shouldContinue ?? { true })
            let embedding = enroller.codesToVoiceEmbedding(codes)
            // Hard guarantee: never write a non-finite embedding. A NaN prefix is
            // continued into every synthesis as runaway babble to the frame cap.
            guard embedding.sum().item(Float.self).isFinite else {
                throw VoxtralTTSError.synthesisError(
                    "Enrollment produced a non-finite voice embedding; refusing to save")
            }
            try MLX.save(arrays: ["embedding": embedding], url: outputURL)
            return embedding
        }
    }

    /// Blend two named voice presets.
    public func blendVoicePresets(_ voiceA: VoxtralVoice, _ voiceB: VoxtralVoice, t: Float) -> MLXArray? {
        guard let embA = voiceEmbeddings[voiceA.rawValue],
              let embB = voiceEmbeddings[voiceB.rawValue] else { return nil }
        return blendVoices(voiceA: embA, voiceB: embB, t: t)
    }

    // MARK: - Streaming Synthesis

    /// Synthesize speech as a stream of audio chunks, enabling real-time playback.
    ///
    /// Each chunk contains decoded PCM audio that can be immediately scheduled on an audio player.
    /// The first chunk measures time-to-first-token (TTFT).
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice preset
    ///   - chunkSize: Number of frames per chunk (1 frame = 80ms audio). Default 10 = 800ms chunks.
    public func synthesizeStreaming(
        text: String,
        voice: VoxtralVoice = .neutralFemale,
        chunkSize: Int = 10,
        seed: UInt64? = nil,
        warmUpText: String? = nil,
        warmUpLeadInFrames: Int = 0
    ) -> AsyncThrowingStream<TTSStreamingChunk, Error> {
        guard let voiceEmb = voiceEmbeddings[voice.rawValue] else {
            return AsyncThrowingStream { $0.finish(throwing: VoxtralTTSError.voiceNotFound("Voice '\(voice.rawValue)' not loaded")) }
        }
        return synthesizeStreaming(text: text, voiceEmbedding: voiceEmb, chunkSize: chunkSize, voiceKey: voice.rawValue,
                                   seed: seed, warmUpText: warmUpText, warmUpLeadInFrames: warmUpLeadInFrames)
    }

    /// Streaming synthesis with an arbitrary `[T, 3072]` voice embedding
    /// (e.g. a cloned voice), mirroring the preset overload above.
    /// Pass a stable `voiceKey` to enable voice-prefix KV caching across calls.
    ///
    /// `seed`, `warmUpText` and `warmUpLeadInFrames` mirror the batch
    /// `synthesize(...)` overload: `seed` makes the flow-matching sampling
    /// reproducible (without it every call draws fresh noise), and `warmUpText`
    /// prepends a short throwaway carrier — a vocalise works best, see
    /// `recommendedWarmUpVocalise` — whose audio is trimmed back off before the
    /// first content chunk is emitted (A6b enrolled-voice stabilization).
    public func synthesizeStreaming(
        text: String,
        voiceEmbedding: MLXArray,
        chunkSize: Int = 10,
        voiceKey: String? = nil,
        seed: UInt64? = nil,
        warmUpText: String? = nil,
        warmUpLeadInFrames: Int = 0
    ) -> AsyncThrowingStream<TTSStreamingChunk, Error> {
        let generation: UInt64
        do {
            generation = try gate.begin(
                "streaming synthesis", accepts: { $0.isReady },
                refusal: VoxtralTTSError.invalidConfiguration("Model not loaded"),
                busy: VoxtralTTSError.busy, state: .synthesizing)
        } catch {
            return AsyncThrowingStream { $0.finish(throwing: error) }
        }
        guard let model = ttsModel, let tokenizer else {
            gate.end(generation, state: .ready)
            return AsyncThrowingStream { $0.finish(throwing: VoxtralTTSError.invalidConfiguration("Model not loaded")) }
        }
        do {
            try model.validateModuleTypes()  // unsupported module types end the stream with an error (K-27)
        } catch {
            gate.end(generation, state: .ready)
            return AsyncThrowingStream { $0.finish(throwing: error) }
        }
        let voiceEmb = voiceEmbedding

        // Reuse (or compute) the voice-conditioned prefix KV when we have a key.
        let prefix: (cache: [any KVCache], len: Int)? = voiceKey.map {
            voicePrefix(model, for: voiceEmb, key: $0)
        }

        let startTime = Date()
        let beacon = RuntimeBeacon.begin(task: "tts-streaming", model: loadedModelID)

        let capturedSampleRate = sampleRate
        let capturedSanitize = configuration.sanitizeText

        // Prepend the warm-up carrier as its own sentence (mirrors the batch
        // synthesize path) so its audio can be located and trimmed below.
        let carrier = warmUpText?.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasWarmUp = !(carrier?.isEmpty ?? true)
        let genText: String
        if let carrier, hasWarmUp {
            let sep = carrier.last.map { ".!?…".contains($0) } == true ? " " : ". "
            genText = carrier + sep + text
        } else {
            genText = text
        }
        // Same text-proportional frame cap as the batch path (K-14)
        let capturedMaxFrames = frameCap(forText: genText)

        // Box non-Sendable captures for Swift 6 strict concurrency
        final class StreamContext: @unchecked Sendable {
            let model: VoxtralTTSModel
            let tokenizer: TekkenTokenizer
            let voiceEmb: MLXArray
            let pipeline: VoxtralTTSPipeline
            let prefixCache: [any KVCache]?
            let prefixLen: Int
            let genText: String
            let seed: UInt64?
            let hasWarmUp: Bool
            let warmUpLeadInFrames: Int
            init(model: VoxtralTTSModel, tokenizer: TekkenTokenizer, voiceEmb: MLXArray, pipeline: VoxtralTTSPipeline, prefixCache: [any KVCache]?, prefixLen: Int, genText: String, seed: UInt64?, hasWarmUp: Bool, warmUpLeadInFrames: Int) {
                self.model = model; self.tokenizer = tokenizer; self.voiceEmb = voiceEmb; self.pipeline = pipeline
                self.prefixCache = prefixCache; self.prefixLen = prefixLen
                self.genText = genText; self.seed = seed; self.hasWarmUp = hasWarmUp; self.warmUpLeadInFrames = warmUpLeadInFrames
            }
        }
        let ctx = StreamContext(model: model, tokenizer: tokenizer, voiceEmb: voiceEmb, pipeline: self, prefixCache: prefix?.cache, prefixLen: prefix?.len ?? 0, genText: genText, seed: seed, hasWarmUp: hasWarmUp, warmUpLeadInFrames: warmUpLeadInFrames)

        // K-12: the producing Task is cancelled when the consumer stops (MLX-003)
        let (stream, continuation) = AsyncThrowingStream<TTSStreamingChunk, Error>.makeStream()
        let task = Task {
            defer { beacon?.end() }
            // Sample offset (into the full decoded waveform) where the real
            // content starts. Without warm-up that's 0; with warm-up it's the
            // carrier cut, located once from the accumulated audio and then
            // held fixed so the carrier is dropped from every emitted chunk.
            var contentStart: Int? = ctx.hasWarmUp ? nil : 0
            let frameSize = capturedSampleRate * 2 / 25  // 80 ms acoustic frame
            var previousContentSamples = 0
            var isFirst = true

            do {
                // The MLX error handler is task-local: the boundary lives in this producing Task (K-1)
                try await withMLXErrors { errors in
                    let codeStream = ctx.model.generateStreaming(
                        text: ctx.genText,
                        voiceEmbedding: ctx.voiceEmb,
                        tokenizer: ctx.tokenizer,
                        maxTokens: capturedMaxFrames,
                        chunkSize: chunkSize,
                        sanitize: capturedSanitize,
                        seed: ctx.seed,
                        prefixCache: ctx.prefixCache,
                        prefixLen: ctx.prefixLen
                    )

                    for try await chunk in codeStream {
                        try Task.checkCancellation()
                        // Decode all accumulated codes to get full waveform
                        let fullWaveform = ctx.model.decodeToWaveform(chunk.accumulatedCodes)
                        MLX.eval(fullWaveform)
                        try errors.check()

                        let totalSamples = fullWaveform.dim(0)

                        // Locate the warm-up carrier's end once, then drop it.
                        // Wait until the adaptive scan window (3 s) has actually
                        // accumulated — deciding on a partial waveform can latch
                        // onto a micro-pause inside the carrier and hold that
                        // wrong cut for the rest of the stream. The batch path
                        // never sees a partial waveform, so it needs no such
                        // guard; this keeps both paths deciding on the same view.
                        let scanWindowSamples = capturedSampleRate * 3
                        if contentStart == nil, totalSamples < scanWindowSamples, !chunk.isFinal {
                            beacon?.update(phase: "streaming", step: chunk.totalFrames, totalSteps: capturedMaxFrames)
                            continue
                        }
                        if contentStart == nil {
                            // Same adaptive cut as the batch path: no absolute
                            // level is assumed. The carrier's terminal pause is
                            // whatever the generation made it — measured −55 dB,
                            // −65 dB and −126 dB across three seeds on one voice
                            // — so a fixed floor finds it only sometimes, and a
                            // peak-relative one rides up with the content and
                            // swallows the carrier. Deriving the reference from
                            // the carrier's own level sidesteps both.
                            let (_, cutFrames) = trimLeadingCarrierAdaptive(
                                fullWaveform, sampleRate: capturedSampleRate,
                                leadInFrames: ctx.warmUpLeadInFrames)
                            if cutFrames > 0 {
                                contentStart = cutFrames * frameSize
                            } else if chunk.isFinal {
                                contentStart = 0  // pause never found — emit everything
                            } else {
                                // Still inside the carrier; nothing to emit yet.
                                beacon?.update(phase: "streaming", step: chunk.totalFrames, totalSteps: capturedMaxFrames)
                                continue
                            }
                        }
                        let start = contentStart!
                        guard totalSamples > start else {
                            beacon?.update(phase: "streaming", step: chunk.totalFrames, totalSteps: capturedMaxFrames)
                            if !chunk.isFinal { continue }
                            // Final chunk with no content past the cut: emit an
                            // empty final marker so consumers see completion.
                            continuation.yield(TTSStreamingChunk(
                                waveform: fullWaveform[(totalSamples)...],
                                frameIndex: chunk.totalFrames, frameCount: 0,
                                totalFrames: chunk.totalFrames, sampleRate: capturedSampleRate,
                                isFirst: isFirst, isFinal: true,
                                elapsed: Date().timeIntervalSince(startTime)))
                            break
                        }

                        // Content samples generated so far, and the new slice.
                        let contentTotal = totalSamples - start
                        let newWaveform: MLXArray
                        if previousContentSamples > 0 && previousContentSamples < contentTotal {
                            newWaveform = fullWaveform[(start + previousContentSamples)...]
                        } else {
                            newWaveform = fullWaveform[start...]
                        }

                        let elapsed = Date().timeIntervalSince(startTime)

                        continuation.yield(TTSStreamingChunk(
                            waveform: newWaveform,
                            frameIndex: chunk.totalFrames - chunk.newFrameCount,
                            frameCount: chunk.newFrameCount,
                            totalFrames: chunk.totalFrames,
                            sampleRate: capturedSampleRate,
                            isFirst: isFirst,
                            isFinal: chunk.isFinal,
                            elapsed: elapsed
                        ))

                        previousContentSamples = contentTotal
                        isFirst = false
                        beacon?.update(phase: "streaming", step: chunk.totalFrames, totalSteps: capturedMaxFrames)
                    }

                    continuation.finish()
                }
            } catch {
                continuation.finish(throwing: error)
            }

            // Ignored when the pipeline was unloaded or reloaded meanwhile (stale Task)
            ctx.pipeline.cachePolicy.endOfResponse()
            ctx.pipeline.gate.end(generation, state: .ready)
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    // MARK: - Resource Management

    public func unload() {
        ttsModel = nil
        tokenizer = nil
        voiceEmbeddings = [:]
        prefixCacheEntry = nil
        modelDirectory = nil
        loadedModelID = nil
        gate.reset(.unloaded)
        cachePolicy.restore()
    }

    public var isReady: Bool { state.isReady }

    public var availableVoices: [VoxtralVoice] {
        VoxtralVoice.allCases.filter { voiceEmbeddings[$0.rawValue] != nil }
    }
}
