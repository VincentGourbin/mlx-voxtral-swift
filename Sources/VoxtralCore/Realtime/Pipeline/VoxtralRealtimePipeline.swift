/**
 * VoxtralRealtimePipeline - High-level API for Voxtral Realtime transcription
 *
 * Usage:
 * ```swift
 * let pipeline = VoxtralRealtimePipeline()
 * try await pipeline.loadModel()
 * let text = try await pipeline.transcribe(audio: audioURL)
 * let embeddings = try await pipeline.extractAudioEmbeddings(audio: audioURL)
 * pipeline.unload()
 * ```
 */

import Foundation
import MLX
import MLXProfiler

public class VoxtralRealtimePipeline: @unchecked Sendable {

    // MARK: - Configuration

    public struct Configuration: Sendable {
        public var maxTokens: Int
        public var temperature: Float
        public var transcriptionDelayMs: Int
        /// MLX buffer-cache limit set after loading and restored at `unload()`; nil leaves the host's
        /// process-wide setting alone (K-52)
        public var cacheLimitBytes: Int?

        public static var `default`: Configuration {
            Configuration(maxTokens: 4096, temperature: 0.0, transcriptionDelayMs: 480)
        }

        public init(maxTokens: Int = 4096, temperature: Float = 0.0, transcriptionDelayMs: Int = 480, cacheLimitBytes: Int? = nil) {
            self.cacheLimitBytes = cacheLimitBytes
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.transcriptionDelayMs = transcriptionDelayMs
        }
    }

    // MARK: - State

    public enum State: Sendable {
        case unloaded, loading, ready, processing, error(String)

        var isUnloaded: Bool { if case .unloaded = self { return true }; return false }
        var isReady: Bool { if case .ready = self { return true }; return false }
    }

    // MARK: - Properties

    public var configuration: Configuration
    /// Read-only view of the gate: every transition happens under its lock (K-11)
    public var state: State { gate.state }

    /// State, running operation and generation token, changed atomically
    let gate = PipelineGate<State>(.unloaded)

    /// Opt-in MLX cache limit held while loaded (K-52)
    private let cachePolicy = MLXCachePolicy()
    public let sampleRate: Int = 16000

    private var model: VoxtralRealtimeModel?
    private var tokenizer: TekkenTokenizer?
    private var modelDirectory: URL?

    public typealias ProgressCallback = @Sendable (Double, String) -> Void

    // MARK: - Initialization

    /// Registry entry of `modelId`: nil → the default model, an unknown id → an error instead of silently loading the
    /// default (K-9)
    static func modelInfo(for modelId: String?) throws -> VoxtralRealtimeModelInfo {
        guard let modelId else { return VoxtralRealtimeRegistry.defaultModel }
        guard let info = VoxtralRealtimeRegistry.model(withId: modelId) else {
            throw VoxtralRealtimeError.invalidConfiguration(
                "Unknown Realtime model id \(modelId); known: \(VoxtralRealtimeRegistry.models.map(\.id).joined(separator: ", "))")
        }
        return info
    }

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    // MARK: - Model Loading

    public func loadModel(
        modelId: String? = nil,
        progress: ProgressCallback? = nil
    ) async throws {
        // MLX errors become VoxtralError.mlx instead of terminating the host (K-1)
        try await withMLXErrors { _ in
            // Atomic check-and-set: a concurrent load or a running operation is refused (K-11)
            let generation = try gate.begin(
                "loading", accepts: { current in current.isUnloaded || { if case .error = current { return true }; return false }() },
                refusal: VoxtralRealtimeError.invalidConfiguration("Model already loaded or loading"),
                busy: VoxtralRealtimeError.busy, state: .loading, newGeneration: true)

            let modelInfo = try Self.modelInfo(for: modelId)
            let beacon = VoxtralRuntimeBeacon.begin(task: "load-realtime-model", model: modelInfo.id)
            defer { beacon?.end() }

            do {
                let session = MLXProfiler.shared.activeSession

                progress?(0.05, "Resolving Realtime model...")
                session?.beginPhase("1. Model Download", category: .modelLoad)
                let modelDir = try await VoxtralModelDownloader.downloadRealtimeModel(modelInfo) { p, msg in
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

                progress?(0.40, "Loading Realtime model...")
                session?.beginPhase("2. Model Loading", category: .modelLoad)
                // Weights load off the cooperative pool (K-15); MLX errors caught on that queue (K-1)
                let loadedModel = try await runOffCooperativePool {
                    try withMLXErrors { _ in
                        try loadVoxtralRealtimeModel(from: modelDir) { p, msg in progress?(0.40 + Double(p) * 0.40, msg) }
                    }
                }
                self.model = loadedModel
                session?.endPhase("2. Model Loading", category: .modelLoad)

                progress?(1.0, "Realtime model ready")
                cachePolicy.apply(configuration.cacheLimitBytes)
                gate.end(generation, state: .ready)

            } catch {
                gate.end(generation, state: .error(error.localizedDescription))
                throw error
            }
        }
    }

    // MARK: - Transcription

    public func transcribe(audio: URL) async throws -> String {
        // MLX errors become VoxtralError.mlx instead of terminating the host (K-1)
        // Off the cooperative pool, cancellable at every step (K-15)
        return try await runOffCooperativePool { [self] in try withMLXErrors { _ in
            let generation = try gate.begin(
                "transcription", accepts: { $0.isReady }, refusal: VoxtralRealtimeError.invalidConfiguration("Model not loaded"),
                busy: VoxtralRealtimeError.busy, state: .processing)
            defer {
                cachePolicy.endOfResponse()
                gate.end(generation, state: .ready)
            }
            guard let model, let tokenizer else {
                throw VoxtralRealtimeError.invalidConfiguration("Model not loaded")
            }

            let session = MLXProfiler.shared.activeSession
            let beacon = VoxtralRuntimeBeacon.begin(task: "transcribe-realtime")
            defer { beacon?.end() }

            do {
                session?.beginPhase("Mel Spectrogram", category: .melSpectrogram)
                let mel = try prepareMel(from: audio, config: model.config)
                session?.endPhase("Mel Spectrogram", category: .melSpectrogram)
                try VoxtralCancellation.check()

                // Generate transcription
                session?.beginPhase("Realtime Generation", category: .generation)
                let (tokens, _) = model.generate(
                    mel: mel,
                    tokenizer: tokenizer,
                    maxTokens: configuration.maxTokens,
                    temperature: configuration.temperature,
                    delayMs: configuration.transcriptionDelayMs
                )
                session?.endPhase("Realtime Generation", category: .generation)
                try VoxtralCancellation.check()  // generation stopped early for a cancelled caller (K-15)
                // Steps whose token carries no text: control tokens ([STREAMING_PAD], [STREAMING_WORD]…),
                // which `decode` skips (K-13)
                let silent = Set(Set(tokens).filter { tokenizer.decode([$0]).isEmpty })
                lastPadFraction = tokens.isEmpty ? nil : Double(tokens.filter { silent.contains($0) }.count) / Double(tokens.count)
                let streamingPad = Int32(model.config.tekkenStreamingPadTokenId)
                lastStreamingPadFraction = tokens.isEmpty
                    ? nil : Double(tokens.filter { $0 == streamingPad }.count) / Double(tokens.count)

                session?.beginPhase("Token Decoding", category: .decoding)
                let text = tokenizer.decode(tokens).trimmingCharacters(in: .whitespacesAndNewlines)
                session?.endPhase("Token Decoding", category: .decoding)
                return text
            }
        }
        }
    }

    // MARK: - Audio Embedding Extraction

    /// Extract audio embeddings from an audio file.
    /// Returns embeddings of shape [1, n_tokens, 3072].
    public func extractAudioEmbeddings(audio: URL) async throws -> MLXArray {
        // MLX errors become VoxtralError.mlx instead of terminating the host (K-1)
        // Off the cooperative pool, cancellable at every step (K-15)
        return try await runOffCooperativePool { [self] in try withMLXErrors { _ in
            let generation = try gate.begin(
                "embedding extraction", accepts: { $0.isReady },
                refusal: VoxtralRealtimeError.invalidConfiguration("Model not loaded"), busy: VoxtralRealtimeError.busy)
            defer { gate.end(generation) }
            guard let model else {
                throw VoxtralRealtimeError.invalidConfiguration("Model not loaded")
            }

            let session = MLXProfiler.shared.activeSession

            session?.beginPhase("Mel Spectrogram", category: .melSpectrogram)
            let mel = try prepareMel(from: audio, config: model.config)
            session?.endPhase("Mel Spectrogram", category: .melSpectrogram)

            session?.beginPhase("Audio Encoding", category: .audioEncode)
            let embeddings = model.extractAudioEmbeddings(mel)
            MLX.eval(embeddings)
            session?.endPhase("Audio Encoding", category: .audioEncode)

            return embeddings
        }
        }
    }

    // MARK: - Resource Management

    public func unload() {
        model = nil
        tokenizer = nil
        modelDirectory = nil
        gate.reset(.unloaded)
        cachePolicy.restore()
    }

    public var isReady: Bool { state.isReady }

    /// True when the last `transcribe` stopped on its text budget (`maxTokens`) before the audio ended (K-5)
    public var lastTranscriptionTruncated: Bool { model?.lastGenerationTruncated ?? false }

    /// Share of the last transcription's decode steps whose token carries no text: control tokens, [STREAMING_PAD]
    /// and [STREAMING_WORD] included (`docs/bench.schema.json`; bench, K-36)
    public private(set) var lastPadFraction: Double?

    /// Share of the last transcription's decode steps whose token is [STREAMING_PAD] alone (K-36, input of K-73)
    public private(set) var lastStreamingPadFraction: Double?

    // MARK: - Audio Preparation

    /// Prepare mel spectrogram with streaming padding protocol.
    /// Pads audio with silence on both sides, computes mel, ensures even frame count.
    private func prepareMel(from audioURL: URL, config: VoxtralRealtimeConfiguration) throws -> MLXArray {
        let audioData = try loadAudio(audioURL.path)

        let nDelay = config.numDelayTokens(delayMs: configuration.transcriptionDelayMs)
        let nLeft = config.nLeftPadTokens
        let nRight = nDelay + 1 + 10
        let rawLen = config.rawAudioLengthPerToken  // 1280

        // Pad audio: left silence + audio + alignment + right silence
        let nSamples = audioData.dim(0)
        let alignPad = (rawLen - (nSamples % rawLen)) % rawLen
        let leftPad = nLeft * rawLen
        let rightPad = alignPad + nRight * rawLen

        let padded = MLX.concatenated([
            MLX.zeros([leftPad]),
            audioData,
            MLX.zeros([rightPad])
        ])

        // Compute mel spectrogram with fixed global max
        var (mel, _) = logMelSpectrogram(
            padded,
            globalMax: config.audioEncoding.globalLogMelMax
        )

        // Ensure even frame count (drop first frame if odd)
        if mel.dim(1) % 2 != 0 {
            mel = mel[0..., 1...]
        }

        return mel
    }
}
