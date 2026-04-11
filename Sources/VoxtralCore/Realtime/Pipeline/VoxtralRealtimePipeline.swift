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

@available(macOS 14.0, *)
public class VoxtralRealtimePipeline: @unchecked Sendable {

    // MARK: - Configuration

    public struct Configuration: Sendable {
        public var maxTokens: Int
        public var temperature: Float
        public var transcriptionDelayMs: Int

        public static var `default`: Configuration {
            Configuration(maxTokens: 4096, temperature: 0.0, transcriptionDelayMs: 480)
        }

        public init(maxTokens: Int = 4096, temperature: Float = 0.0, transcriptionDelayMs: Int = 480) {
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
    public private(set) var state: State = .unloaded
    public let sampleRate: Int = 16000

    private var model: VoxtralRealtimeModel?
    private var tokenizer: TekkenTokenizer?
    private var modelDirectory: URL?

    public typealias ProgressCallback = @Sendable (Double, String) -> Void

    // MARK: - Initialization

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    // MARK: - Model Loading

    public func loadModel(
        modelId: String? = nil,
        progress: ProgressCallback? = nil
    ) async throws {
        guard state.isUnloaded || { if case .error = state { return true }; return false }() else {
            throw VoxtralRealtimeError.invalidConfiguration("Model already loaded or loading")
        }

        state = .loading

        do {
            let session = MLXProfiler.shared.activeSession

            progress?(0.05, "Resolving Realtime model...")
            session?.beginPhase("1. Model Download", category: .modelLoad)
            let modelInfo = modelId.flatMap { VoxtralRealtimeRegistry.model(withId: $0) }
                ?? VoxtralRealtimeRegistry.defaultModel
            let modelDir = try await ModelDownloader.downloadRealtimeModel(modelInfo) { p, msg in
                progress?(0.05 + p * 0.35, msg)
            }
            self.modelDirectory = modelDir
            session?.endPhase("1. Model Download", category: .modelLoad)

            progress?(0.40, "Loading Realtime model...")
            session?.beginPhase("2. Model Loading", category: .modelLoad)
            let loadedModel = try loadVoxtralRealtimeModel(from: modelDir) { p, msg in
                progress?(0.40 + Double(p) * 0.40, msg)
            }
            self.model = loadedModel
            session?.endPhase("2. Model Loading", category: .modelLoad)

            progress?(0.85, "Loading tokenizer...")
            session?.beginPhase("3. Tokenizer Loading", category: .tokenization)
            self.tokenizer = TekkenTokenizer(modelPath: modelDir.path)
            session?.endPhase("3. Tokenizer Loading", category: .tokenization)

            progress?(1.0, "Realtime model ready")
            state = .ready

        } catch {
            state = .error(error.localizedDescription)
            throw error
        }
    }

    // MARK: - Transcription

    public func transcribe(audio: URL) async throws -> String {
        guard state.isReady, let model, let tokenizer else {
            throw VoxtralRealtimeError.invalidConfiguration("Model not loaded")
        }

        state = .processing
        let session = MLXProfiler.shared.activeSession

        do {
            session?.beginPhase("Mel Spectrogram", category: .melSpectrogram)
            let mel = try prepareMel(from: audio, config: model.config)
            session?.endPhase("Mel Spectrogram", category: .melSpectrogram)

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

            session?.beginPhase("Token Decoding", category: .decoding)
            let text = tokenizer.decode(tokens).trimmingCharacters(in: .whitespacesAndNewlines)
            session?.endPhase("Token Decoding", category: .decoding)
            state = .ready
            return text

        } catch {
            state = .ready
            throw error
        }
    }

    // MARK: - Audio Embedding Extraction

    /// Extract audio embeddings from an audio file.
    /// Returns embeddings of shape [1, n_tokens, 3072].
    public func extractAudioEmbeddings(audio: URL) async throws -> MLXArray {
        guard state.isReady, let model else {
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

    // MARK: - Resource Management

    public func unload() {
        model = nil
        tokenizer = nil
        modelDirectory = nil
        state = .unloaded
    }

    public var isReady: Bool { state.isReady }

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
