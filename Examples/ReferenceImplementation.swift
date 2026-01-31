/**
 * Reference Implementation - VoxtralCore Framework
 *
 * This is the EXACT way to use VoxtralPipeline for transcription and chat.
 * Copy this pattern in your app.
 *
 * Tested with: v1.0.8, mini-3b-4bit, hybrid backend
 */

import Foundation
import VoxtralCore

// MARK: - Minimal Service Example

@MainActor
final class VoxtralReferenceService {

    private var pipeline: VoxtralPipeline?
    private(set) var isLoaded = false

    // MARK: - Configuration

    /// Model to use - change as needed
    static let modelVariant: VoxtralPipeline.Model = .mini3b4bit

    /// Backend - .hybrid uses Core ML encoder + MLX decoder
    static let backend: VoxtralPipeline.Backend = .hybrid

    // MARK: - Load Model

    /// Load the model - call this ONCE before any transcription/chat
    func loadModel() async throws {
        guard !isLoaded else {
            print("[Reference] Model already loaded")
            return
        }

        print("[Reference] Creating pipeline...")

        // Create pipeline with default configuration
        let newPipeline = VoxtralPipeline(
            model: Self.modelVariant,
            backend: Self.backend
        )

        print("[Reference] Loading model (this may download on first run)...")

        // Load model with progress callback
        try await newPipeline.loadModel { progress, status in
            print("[Reference] \(Int(progress * 100))% - \(status)")
        }

        self.pipeline = newPipeline
        self.isLoaded = true

        print("[Reference] Model loaded successfully!")
    }

    // MARK: - Transcribe

    /// Transcribe audio file to text
    func transcribe(audioURL: URL) async throws -> String {
        guard let pipeline = pipeline, isLoaded else {
            throw NSError(domain: "VoxtralReference", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }

        print("[Reference] Transcribing: \(audioURL.lastPathComponent)")

        let result = try await pipeline.transcribe(audio: audioURL)

        print("[Reference] Transcription complete (\(result.count) chars)")
        return result
    }

    // MARK: - Chat

    /// Chat with audio context
    func chat(audioURL: URL, prompt: String) async throws -> String {
        guard let pipeline = pipeline, isLoaded else {
            throw NSError(domain: "VoxtralReference", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }

        print("[Reference] Chat with audio: \(audioURL.lastPathComponent)")
        print("[Reference] Prompt: \(prompt.prefix(50))...")

        let result = try await pipeline.chat(audio: audioURL, prompt: prompt)

        print("[Reference] Chat complete (\(result.count) chars)")
        return result
    }

    // MARK: - Unload

    /// Unload model to free memory
    func unloadModel() {
        pipeline?.unload()
        pipeline = nil
        isLoaded = false
        print("[Reference] Model unloaded")
    }
}

// MARK: - Usage Example

/*
 Usage in your app:

 ```swift
 let service = VoxtralReferenceService()

 // 1. Load model (do this once, e.g., at app startup or on-demand)
 try await service.loadModel()

 // 2. Transcribe or chat
 let text = try await service.transcribe(audioURL: myAudioFile)
 // or
 let response = try await service.chat(audioURL: myAudioFile, prompt: "Summarize this")

 // 3. Unload when done (optional, frees memory)
 service.unloadModel()
 ```

 IMPORTANT:
 - Do NOT use Task.detached for loadModel() - it may cause threading issues
 - Call loadModel() only ONCE, check isLoaded before calling again
 - Use @MainActor for the service to avoid concurrency issues
 */
