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

@available(macOS 14.0, *)
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

        public static var `default`: Configuration {
            Configuration(maxFrames: 2500, temperature: 0.0, cfgAlpha: 1.2, flowSteps: 8, sanitizeText: true, trimLeadIn: true, trimTail: false)
        }

        public init(maxFrames: Int = 2500, temperature: Float = 0.0, cfgAlpha: Float = 1.2, flowSteps: Int = 8, sanitizeText: Bool = true, trimLeadIn: Bool = true, trimTail: Bool = false) {
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
    public private(set) var state: State = .unloaded
    public let sampleRate: Int = 24000

    private var ttsModel: VoxtralTTSModel?
    private var tokenizer: TekkenTokenizer?
    private let voiceManager: VoxtralVoicePresetManager
    private var modelDirectory: URL?
    private var voiceEmbeddings: [String: MLXArray] = [:]
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
        guard state.isUnloaded || { if case .error = state { return true }; return false }() else {
            throw VoxtralTTSError.invalidConfiguration("Model already loaded or loading")
        }

        state = .loading
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

            progress?(0.40, "Loading TTS model...")
            session?.beginPhase("2. Model Loading", category: .modelLoad)
            let model = try loadVoxtralTTSModel(from: modelDir) { p, msg in
                progress?(0.40 + Double(p) * 0.40, msg)
            }
            self.ttsModel = model
            session?.endPhase("2. Model Loading", category: .modelLoad)

            progress?(0.85, "Loading tokenizer...")
            session?.beginPhase("3. Tokenizer Loading", category: .tokenization)
            // TekkenTokenizer expects the MODEL DIRECTORY, not the tekken.json file path
            self.tokenizer = TekkenTokenizer(modelPath: modelDir.path)
            session?.endPhase("3. Tokenizer Loading", category: .tokenization)

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
            state = .ready

        } catch {
            state = .error(error.localizedDescription)
            throw error
        }
    }

    // MARK: - Synthesis

    public func synthesize(
        text: String,
        voice: VoxtralVoice = .neutralFemale,
        seed: UInt64? = nil
    ) async throws -> TTSSynthesisResult {
        guard state.isReady, let model = ttsModel, let tokenizer else {
            throw VoxtralTTSError.invalidConfiguration("Model not loaded")
        }

        guard let voiceEmb = voiceEmbeddings[voice.rawValue] else {
            throw VoxtralTTSError.voiceNotFound("Voice '\(voice.rawValue)' not loaded")
        }

        state = .synthesizing
        let startTime = Date()
        let profiler = MLXProfiler.shared
        let session = profiler.activeSession
        let beacon = RuntimeBeacon.begin(task: "tts", model: loadedModelID)
        defer { beacon?.end() }

        let prefix = voicePrefix(model, for: voiceEmb, key: voice.rawValue)
        do {
            // Generate audio codes (semantic code generation + flow matching inside)
            profiler.startSemanticGen()
            let (codes, numFrames, ttft) = model.generate(
                text: text,
                voiceEmbedding: voiceEmb,
                tokenizer: tokenizer,
                maxTokens: configuration.maxFrames,
                sanitize: configuration.sanitizeText,
                seed: seed,
                prefixCache: prefix.cache,
                prefixLen: prefix.len
            )
            profiler.endSemanticGen(frameCount: numFrames)
            profiler.setTTFT(ttft)

            guard numFrames > 0 else {
                state = .ready
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
            state = .ready

            return TTSSynthesisResult(
                waveform: waveform,
                numFrames: numFrames,
                sampleRate: sampleRate,
                generationTime: generationTime,
                timeToFirstToken: ttft
            )
        } catch {
            state = .ready
            throw error
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
        guard state.isReady, let model = ttsModel, let tokenizer else {
            throw VoxtralTTSError.invalidConfiguration("Model not loaded")
        }

        state = .synthesizing
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
            let (codes, numFrames, ttft) = model.generate(
                text: genText,
                voiceEmbedding: voiceEmbedding,
                tokenizer: tokenizer,
                maxTokens: configuration.maxFrames,
                sanitize: configuration.sanitizeText,
                seed: seed
            )
            profiler.endSemanticGen(frameCount: numFrames)
            profiler.setTTFT(ttft)

            guard numFrames > 0 else {
                state = .ready
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
                ? trimLeadingCarrier(rawWaveform, sampleRate: sampleRate,
                                     relativeThresholdDB: -100, absoluteFloor: 4e-4,
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
            state = .ready

            return TTSSynthesisResult(
                waveform: waveform,
                numFrames: numFrames,
                sampleRate: sampleRate,
                generationTime: generationTime,
                timeToFirstToken: ttft
            )
        } catch {
            state = .ready
            throw error
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
        guard state.isReady, let model = ttsModel else {
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
        chunkSize: Int = 10
    ) -> AsyncThrowingStream<TTSStreamingChunk, Error> {
        guard let voiceEmb = voiceEmbeddings[voice.rawValue] else {
            return AsyncThrowingStream { $0.finish(throwing: VoxtralTTSError.voiceNotFound("Voice '\(voice.rawValue)' not loaded")) }
        }
        return synthesizeStreaming(text: text, voiceEmbedding: voiceEmb, chunkSize: chunkSize, voiceKey: voice.rawValue)
    }

    /// Streaming synthesis with an arbitrary `[T, 3072]` voice embedding
    /// (e.g. a cloned voice), mirroring the preset overload above.
    /// Pass a stable `voiceKey` to enable voice-prefix KV caching across calls.
    public func synthesizeStreaming(
        text: String,
        voiceEmbedding: MLXArray,
        chunkSize: Int = 10,
        voiceKey: String? = nil
    ) -> AsyncThrowingStream<TTSStreamingChunk, Error> {
        guard state.isReady, let model = ttsModel, let tokenizer else {
            return AsyncThrowingStream { $0.finish(throwing: VoxtralTTSError.invalidConfiguration("Model not loaded")) }
        }
        let voiceEmb = voiceEmbedding

        // Reuse (or compute) the voice-conditioned prefix KV when we have a key.
        let prefix: (cache: [any KVCache], len: Int)? = voiceKey.map {
            voicePrefix(model, for: voiceEmb, key: $0)
        }

        state = .synthesizing
        let startTime = Date()
        let beacon = RuntimeBeacon.begin(task: "tts-streaming", model: loadedModelID)

        let capturedMaxFrames = configuration.maxFrames
        let capturedSampleRate = sampleRate
        let capturedSanitize = configuration.sanitizeText

        // Box non-Sendable captures for Swift 6 strict concurrency
        final class StreamContext: @unchecked Sendable {
            let model: VoxtralTTSModel
            let tokenizer: TekkenTokenizer
            let voiceEmb: MLXArray
            let pipeline: VoxtralTTSPipeline
            let prefixCache: [any KVCache]?
            let prefixLen: Int
            init(model: VoxtralTTSModel, tokenizer: TekkenTokenizer, voiceEmb: MLXArray, pipeline: VoxtralTTSPipeline, prefixCache: [any KVCache]?, prefixLen: Int) {
                self.model = model; self.tokenizer = tokenizer; self.voiceEmb = voiceEmb; self.pipeline = pipeline
                self.prefixCache = prefixCache; self.prefixLen = prefixLen
            }
        }
        let ctx = StreamContext(model: model, tokenizer: tokenizer, voiceEmb: voiceEmb, pipeline: self, prefixCache: prefix?.cache, prefixLen: prefix?.len ?? 0)

        return AsyncThrowingStream { continuation in
            Task {
                defer { beacon?.end() }
                var previousSampleCount = 0
                var isFirst = true

                do {
                    let codeStream = ctx.model.generateStreaming(
                        text: text,
                        voiceEmbedding: ctx.voiceEmb,
                        tokenizer: ctx.tokenizer,
                        maxTokens: capturedMaxFrames,
                        chunkSize: chunkSize,
                        sanitize: capturedSanitize,
                        prefixCache: ctx.prefixCache,
                        prefixLen: ctx.prefixLen
                    )

                    for try await chunk in codeStream {
                        // Decode all accumulated codes to get full waveform
                        let fullWaveform = ctx.model.decodeToWaveform(chunk.accumulatedCodes)
                        MLX.eval(fullWaveform)

                        let totalSamples = fullWaveform.dim(0)

                        // Extract only the new samples
                        let newWaveform: MLXArray
                        if previousSampleCount > 0 && previousSampleCount < totalSamples {
                            newWaveform = fullWaveform[previousSampleCount...]
                        } else {
                            newWaveform = fullWaveform
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

                        previousSampleCount = totalSamples
                        isFirst = false
                        beacon?.update(phase: "streaming", step: chunk.totalFrames, totalSteps: capturedMaxFrames)
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }

                ctx.pipeline.state = .ready
            }
        }
    }

    // MARK: - Resource Management

    public func unload() {
        ttsModel = nil
        tokenizer = nil
        voiceEmbeddings = [:]
        prefixCacheEntry = nil
        modelDirectory = nil
        loadedModelID = nil
        state = .unloaded
    }

    public var isReady: Bool { state.isReady }

    public var availableVoices: [VoxtralVoice] {
        VoxtralVoice.allCases.filter { voiceEmbeddings[$0.rawValue] != nil }
    }
}
