/**
 * VoxtralTranscriptionManager - High-level API for Voxtral transcription
 *
 * Simplified facade for Fluxforge Studio and other integrations.
 * Wraps VoxtralPipeline with an even simpler API focused on the default model.
 *
 * Usage:
 * ```swift
 * let manager = VoxtralTranscriptionManager()
 * try await manager.loadModel()
 * let result = try await manager.transcribe(audioURL: audioFile)
 * print(result.text)
 * manager.unloadModel()
 * ```
 */

import Foundation
import MLX

// MARK: - TranscriptionResult

/// Result of a transcription operation
public struct TranscriptionResult: Sendable {
    /// The transcribed text
    public let text: String

    /// Number of tokens generated
    public let tokenCount: Int

    /// Duration of generation in seconds
    public let duration: TimeInterval

    /// Tokens per second
    public var tokensPerSecond: Double {
        guard duration > 0 else { return 0 }
        return Double(tokenCount) / duration
    }

    public init(text: String, tokenCount: Int = 0, duration: TimeInterval = 0) {
        self.text = text
        self.tokenCount = tokenCount
        self.duration = duration
    }
}

// MARK: - VoxtralTranscriptionManager

/// High-level API for Voxtral transcription
/// Thread-safe wrapper around VoxtralPipeline for easy integration
@available(macOS 13.0, iOS 16.0, *)
@MainActor
public class VoxtralTranscriptionManager: @unchecked Sendable {

    // MARK: - Properties

    /// Underlying pipeline instance
    private var pipeline: VoxtralPipeline?

    /// Current model variant
    private var modelVariant: VoxtralPipeline.Model

    /// Whether model is loaded
    public var isLoaded: Bool {
        pipeline?.isReady ?? false
    }

    /// Current memory usage summary
    public var memorySummary: String {
        pipeline?.memorySummary ?? "No model loaded"
    }

    // MARK: - Initialization

    /// Create a new transcription manager with default model (mini-3b-8bit)
    public init() {
        self.modelVariant = .mini3b8bit
    }

    /// Create a transcription manager with a specific model
    /// - Parameter model: The model variant to use
    public init(model: VoxtralPipeline.Model) {
        self.modelVariant = model
    }

    // MARK: - Model Management

    /// Load the default model
    /// Downloads if needed, then loads into memory
    /// - Parameter progress: Optional progress callback (0.0-1.0, status message)
    public func loadModel(progress: (@Sendable (Double, String) -> Void)? = nil) async throws {
        pipeline = VoxtralPipeline(
            model: modelVariant,
            backend: .auto,
            configuration: .default
        )

        try await pipeline?.loadModel(progress: progress)
    }

    /// Unload model to free memory
    public func unloadModel() {
        pipeline?.unload()
        pipeline = nil
    }

    // MARK: - Transcription

    /// Transcribe audio file
    /// - Parameters:
    ///   - audioURL: URL to the audio file
    ///   - language: Language code (default: "en")
    /// - Returns: Transcription result with text and metadata
    public func transcribe(audioURL: URL, language: String = "en") async throws -> TranscriptionResult {
        guard let pipeline = pipeline, isLoaded else {
            throw VoxtralTranscriptionError.modelNotLoaded
        }

        let startTime = Date()
        let text = try await pipeline.transcribe(audio: audioURL, language: language)
        let duration = Date().timeIntervalSince(startTime)

        return TranscriptionResult(
            text: text,
            tokenCount: 0, // Pipeline doesn't expose this directly
            duration: duration
        )
    }

    // MARK: - Chat

    /// Chat with the model using audio context
    /// - Parameters:
    ///   - audioURL: URL to the audio file
    ///   - prompt: Question or instruction about the audio
    ///   - language: Language code (default: "en")
    /// - Returns: Model's response
    public func chat(audioURL: URL, prompt: String, language: String = "en") async throws -> String {
        guard let pipeline = pipeline, isLoaded else {
            throw VoxtralTranscriptionError.modelNotLoaded
        }

        return try await pipeline.chat(audio: audioURL, prompt: prompt, language: language)
    }

    /// Chat with system prompt (text-only, for post-processing transcriptions)
    /// - Parameters:
    ///   - systemPrompt: System instructions
    ///   - userMessage: User's message (e.g., a transcription to summarize)
    /// - Returns: Model's response
    public func chat(systemPrompt: String, userMessage: String) async throws -> String {
        // For text-only chat, we need the full model
        // This is a simplified version - for full text chat, use a dedicated LLM
        throw VoxtralTranscriptionError.audioRequired
    }
}

// MARK: - Static Convenience Methods

@available(macOS 13.0, iOS 16.0, *)
extension VoxtralTranscriptionManager {

    /// Check if the default model is downloaded
    public static func isDefaultModelDownloaded() -> Bool {
        return ModelDownloader.findModelPath(for: ModelRegistry.defaultModel) != nil
    }

    /// Check if a specific model variant is downloaded
    public static func isModelDownloaded(_ model: VoxtralPipeline.Model) -> Bool {
        guard let info = ModelRegistry.model(withId: model.rawValue) else {
            return false
        }
        return ModelDownloader.findModelPath(for: info) != nil
    }

    /// Download the default model
    /// - Parameter progress: Progress callback (0.0-1.0, status message)
    /// - Returns: Path to the downloaded model
    @discardableResult
    public static func downloadDefaultModel(
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        return try await ModelDownloader.download(ModelRegistry.defaultModel, progress: progress)
    }

    /// Delete the default model
    public static func deleteDefaultModel() throws {
        try ModelDownloader.deleteModel(ModelRegistry.defaultModel)
    }

    /// Get info about the default model
    public static var defaultModelInfo: VoxtralModelInfo {
        ModelRegistry.defaultModel
    }

    /// Get all available models
    public static var availableModels: [VoxtralModelInfo] {
        ModelRegistry.models
    }

    /// Recommended model for the current system
    public static func recommendedModel(forRAMGB ramGB: Int? = nil) -> VoxtralPipeline.Model {
        VoxtralPipeline.recommendedModel(forRAMGB: ramGB)
    }
}

// MARK: - Errors

/// Transcription manager specific errors
public enum VoxtralTranscriptionError: Error, LocalizedError {
    case modelNotLoaded
    case audioRequired
    case transcriptionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model not loaded. Call loadModel() first."
        case .audioRequired:
            return "This operation requires audio input. Use the audio-based chat method instead."
        case .transcriptionFailed(let reason):
            return "Transcription failed: \(reason)"
        }
    }
}
