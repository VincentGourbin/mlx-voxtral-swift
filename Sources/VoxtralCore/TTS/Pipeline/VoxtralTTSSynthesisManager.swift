/**
 * VoxtralTTSSynthesisManager - Simplified TTS API
 *
 * A minimal wrapper around VoxtralTTSPipeline for quick synthesis tasks.
 * Mirrors VoxtralTranscriptionManager from the STT module.
 *
 * Usage:
 * ```swift
 * let manager = VoxtralTTSSynthesisManager()
 * try await manager.loadModel()
 * let result = try await manager.synthesize(text: "Hello!")
 * print("Generated \(result.duration)s of audio in \(result.generationTime)s")
 * ```
 */

import Foundation
import MLX

@available(macOS 14.0, *)
public class VoxtralTTSSynthesisManager: @unchecked Sendable {

    /// The underlying pipeline
    private let pipeline: VoxtralTTSPipeline

    /// Default voice to use
    public var defaultVoice: VoxtralVoice

    /// Whether the model is loaded
    public var isLoaded: Bool { pipeline.isReady }

    /// Progress callback type
    public typealias ProgressCallback = @Sendable (Double, String) -> Void

    // MARK: - Initialization

    /// Create a new synthesis manager.
    ///
    /// - Parameters:
    ///   - voice: Default voice preset (default: .neutralFemale)
    ///   - configuration: Pipeline configuration
    public init(
        voice: VoxtralVoice = .neutralFemale,
        configuration: VoxtralTTSPipeline.Configuration = .default
    ) {
        self.defaultVoice = voice
        self.pipeline = VoxtralTTSPipeline(configuration: configuration)
    }

    // MARK: - Model Management

    /// Load the TTS model.
    public func loadModel(progress: ProgressCallback? = nil) async throws {
        try await pipeline.loadModel(progress: progress)
    }

    /// Unload the model and free memory.
    public func unloadModel() {
        pipeline.unload()
    }

    // MARK: - Synthesis

    /// Synthesize speech from text using the default voice.
    ///
    /// - Parameter text: Text to synthesize
    /// - Returns: Synthesis result with waveform and metrics
    public func synthesize(text: String) async throws -> TTSSynthesisResult {
        try await pipeline.synthesize(text: text, voice: defaultVoice)
    }

    /// Synthesize speech with a specific voice.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice preset to use
    /// - Returns: Synthesis result
    public func synthesize(text: String, voice: VoxtralVoice) async throws -> TTSSynthesisResult {
        try await pipeline.synthesize(text: text, voice: voice)
    }

    /// Synthesize and save to a WAV file.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - outputURL: URL for the output WAV file
    ///   - voice: Voice preset (uses default if nil)
    /// - Returns: Synthesis result with metrics
    @discardableResult
    public func synthesizeToFile(
        text: String,
        outputURL: URL,
        voice: VoxtralVoice? = nil
    ) async throws -> TTSSynthesisResult {
        try await pipeline.synthesizeToFile(
            text: text,
            voice: voice ?? defaultVoice,
            outputURL: outputURL
        )
    }

    /// Available voices in the loaded model.
    public var availableVoices: [VoxtralVoice] {
        pipeline.availableVoices
    }

    /// Summary of current state for debugging.
    public var statusSummary: String {
        if isLoaded {
            return "TTS ready | Voice: \(defaultVoice.displayName) | Voices available: \(availableVoices.count)"
        } else {
            return "TTS model not loaded"
        }
    }
}
