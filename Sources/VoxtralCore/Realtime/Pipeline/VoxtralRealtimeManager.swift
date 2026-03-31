/**
 * VoxtralRealtimeManager - Simplified Realtime transcription API
 *
 * A minimal wrapper around VoxtralRealtimePipeline for quick transcription tasks.
 * Mirrors VoxtralTranscriptionManager (STT) and VoxtralTTSSynthesisManager (TTS).
 *
 * Usage:
 * ```swift
 * let manager = VoxtralRealtimeManager()
 * try await manager.loadModel()
 * let text = try await manager.transcribe(audio: audioURL)
 * let embeddings = try await manager.extractEmbeddings(audio: audioURL)
 * manager.unloadModel()
 * ```
 */

import Foundation
import MLX

@available(macOS 14.0, *)
public class VoxtralRealtimeManager: @unchecked Sendable {

    private let pipeline: VoxtralRealtimePipeline

    /// Whether the model is loaded and ready
    public var isLoaded: Bool { pipeline.isReady }

    /// Progress callback type
    public typealias ProgressCallback = @Sendable (Double, String) -> Void

    // MARK: - Initialization

    /// Create a new Realtime manager.
    ///
    /// - Parameters:
    ///   - modelId: Model ID to use (default: recommended 4-bit model)
    ///   - transcriptionDelayMs: Streaming delay in ms (default: 480ms)
    ///   - configuration: Full pipeline configuration (overrides other params)
    public init(
        transcriptionDelayMs: Int = 480,
        configuration: VoxtralRealtimePipeline.Configuration? = nil
    ) {
        if let configuration {
            self.pipeline = VoxtralRealtimePipeline(configuration: configuration)
        } else {
            var config = VoxtralRealtimePipeline.Configuration.default
            config.transcriptionDelayMs = transcriptionDelayMs
            self.pipeline = VoxtralRealtimePipeline(configuration: config)
        }
    }

    // MARK: - Model Management

    /// Load the Realtime model.
    public func loadModel(
        modelId: String? = nil,
        progress: ProgressCallback? = nil
    ) async throws {
        try await pipeline.loadModel(modelId: modelId, progress: progress)
    }

    /// Unload the model and free memory.
    public func unloadModel() {
        pipeline.unload()
    }

    // MARK: - Transcription

    /// Transcribe audio from a file URL.
    public func transcribe(audio: URL) async throws -> String {
        try await pipeline.transcribe(audio: audio)
    }

    // MARK: - Audio Embeddings

    /// Extract audio embeddings from a file URL.
    /// Returns embeddings of shape [1, n_tokens, 3072].
    public func extractEmbeddings(audio: URL) async throws -> MLXArray {
        try await pipeline.extractAudioEmbeddings(audio: audio)
    }

    // MARK: - Status

    /// Summary of current state for debugging.
    public var statusSummary: String {
        if isLoaded {
            return "Realtime ready | Delay: \(pipeline.configuration.transcriptionDelayMs)ms"
        } else {
            return "Realtime model not loaded"
        }
    }
}
