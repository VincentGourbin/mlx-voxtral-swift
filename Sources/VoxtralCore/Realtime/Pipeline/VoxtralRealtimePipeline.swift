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
            progress?(0.05, "Resolving Realtime model...")
            let modelInfo = modelId.flatMap { VoxtralRealtimeRegistry.model(withId: $0) }
                ?? VoxtralRealtimeRegistry.defaultModel
            let modelDir = try await ModelDownloader.downloadRealtimeModel(modelInfo) { p, msg in
                progress?(0.05 + p * 0.35, msg)
            }
            self.modelDirectory = modelDir

            progress?(0.40, "Loading Realtime model...")
            let loadedModel = try loadVoxtralRealtimeModel(from: modelDir) { p, msg in
                progress?(0.40 + Double(p) * 0.40, msg)
            }
            self.model = loadedModel

            progress?(0.85, "Loading tokenizer...")
            self.tokenizer = TekkenTokenizer(modelPath: modelDir.path)

            progress?(0.90, "Ready...")

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

        do {
            // Extract mel spectrogram (variable length, no 30s chunking)
            let audioData = try loadAudio(audio.path)
            let (mel, _) = logMelSpectrogram(
                audioData,
                globalMax: model.config.audioEncoding.globalLogMelMax
            )

            // Generate transcription
            let (tokens, _) = model.generate(
                mel: mel,
                tokenizer: tokenizer,
                maxTokens: configuration.maxTokens,
                temperature: configuration.temperature,
                delayMs: configuration.transcriptionDelayMs
            )

            let text = tokenizer.decode(tokens).trimmingCharacters(in: .whitespacesAndNewlines)
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

        let audioData = try loadAudio(audio.path)
        let (mel, _) = logMelSpectrogram(
            audioData,
            globalMax: model.config.audioEncoding.globalLogMelMax
        )
        let embeddings = model.extractAudioEmbeddings(mel)
        MLX.eval(embeddings)
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
}
