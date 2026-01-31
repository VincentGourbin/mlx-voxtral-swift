/**
 * VoxtralCLI - Command-line interface for Voxtral speech-to-text
 *
 * IMPORTANT: This CLI uses ONLY the high-level VoxtralPipeline API.
 * This ensures the public API works correctly for external users.
 *
 * Features:
 * - List available models
 * - Download models from HuggingFace
 * - Transcribe audio files
 * - Chat mode (ask questions about audio)
 */

import Foundation
import VoxtralCore
import ArgumentParser

@main
struct VoxtralCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voxtral",
        abstract: "Voxtral speech-to-text for Apple Silicon",
        version: "1.1.0",
        subcommands: [
            ListModels.self,
            Download.self,
            Transcribe.self,
            Chat.self
        ],
        defaultSubcommand: Transcribe.self
    )
}

// MARK: - List Models Command

struct ListModels: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List available Voxtral models"
    )

    @Flag(name: .shortAndLong, help: "Show only downloaded models")
    var downloaded = false

    func run() throws {
        if downloaded {
            let downloadedModels = ModelDownloader.listDownloadedModels()
            if downloadedModels.isEmpty {
                print("\nNo models downloaded yet.")
                print("Use 'voxtral download <model-id>' to download a model.")
                print("Use 'voxtral list' to see available models.")
            } else {
                print("\n" + String(repeating: "=", count: 60))
                print("DOWNLOADED MODELS")
                print(String(repeating: "=", count: 60))
                for model in downloadedModels {
                    if let path = ModelDownloader.findModelPath(for: model) {
                        print("\n  \(model.id): \(model.name)")
                        print("    Path: \(path.path)")
                    }
                }
            }
        } else {
            ModelRegistry.printAvailableModels()

            // Also show downloaded status
            print("\nDownload status:")
            for model in ModelRegistry.models {
                let status = ModelDownloader.findModelPath(for: model) != nil ? "[downloaded]" : "[not downloaded]"
                print("  \(model.id): \(status)")
            }
        }
    }
}

// MARK: - Download Command

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Download a model from HuggingFace"
    )

    @Argument(help: "Model ID (e.g., 'mini-3b-8bit') or HuggingFace repo (e.g., 'mzbac/voxtral-mini-3b-8bit')")
    var model: String

    func run() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VOXTRAL MODEL DOWNLOAD")
        print(String(repeating: "=", count: 60))

        let modelPath = try await ModelDownloader.resolveModel(model) { progress, message in
            print("[\(Int(progress * 100))%] \(message)")
        }

        print("\n" + String(repeating: "=", count: 60))
        print("Download complete!")
        print("Model path: \(modelPath.path)")
        print("\nTo transcribe audio:")
        print("  voxtral transcribe --model \(model) <audio-file>")
        print(String(repeating: "=", count: 60))
    }
}

// MARK: - Transcribe Command (Uses VoxtralPipeline ONLY)

struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe an audio file using VoxtralPipeline API"
    )

    @Argument(help: "Path to audio file (MP3, WAV, M4A, etc.)")
    var audioFile: String

    @Option(name: .shortAndLong, help: "Model: mini-3b, mini-3b-8bit, mini-3b-4bit, small-24b, small-24b-8bit, small-4bit")
    var model: String = "mini-3b-8bit"

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 500

    @Option(name: [.customShort("l"), .long], help: "Language code (e.g., 'en', 'fr')")
    var language: String = "en"

    @Option(name: .shortAndLong, help: "Backend: 'mlx' (GPU only) or 'hybrid' (Core ML encoder + MLX decoder)")
    var backend: String = "mlx"

    func run() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VOXTRAL TRANSCRIPTION (VoxtralPipeline API)")
        print(String(repeating: "=", count: 60))

        // Validate audio file
        let audioURL = URL(fileURLWithPath: audioFile)
        guard FileManager.default.fileExists(atPath: audioFile) else {
            throw ValidationError("Audio file not found: \(audioFile)")
        }

        // Parse model option
        guard let pipelineModel = parseModel(model) else {
            throw ValidationError("Unknown model: \(model). Use 'voxtral list' to see available models.")
        }

        // Parse backend option
        let pipelineBackend: VoxtralPipeline.Backend = backend.lowercased() == "hybrid" ? .hybrid : .mlx
        print("\nModel: \(pipelineModel.displayName)")
        print("Backend: \(pipelineBackend.displayName)")

        // Create pipeline with high-level API
        var config = VoxtralPipeline.Configuration.default
        config.maxTokens = maxTokens
        config.temperature = 0.0  // Deterministic for transcription

        let pipeline = VoxtralPipeline(
            model: pipelineModel,
            backend: pipelineBackend,
            configuration: config
        )

        // Load model
        print("\n[1/3] Loading model...")
        let startLoad = Date()
        try await pipeline.loadModel { progress, status in
            print("  [\(Int(progress * 100))%] \(status)")
        }
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Model loaded in \(String(format: "%.2f", loadTime))s")
        print("  Encoder: \(pipeline.encoderStatus)")

        // Transcribe
        print("\n[2/3] Processing audio: \(audioFile)")
        print("\n[3/3] Generating transcription...")
        let startGen = Date()

        let transcription = try await pipeline.transcribe(audio: audioURL, language: language)

        let genTime = Date().timeIntervalSince(startGen)

        // Output result
        print("\n" + String(repeating: "-", count: 60))
        print(transcription)
        print(String(repeating: "-", count: 60))

        // Statistics
        let tokenCount = transcription.split(separator: " ").count  // Approximate
        print("\nStatistics:")
        print("  Time: \(String(format: "%.2f", genTime))s")
        print("  Memory: \(pipeline.memorySummary)")

        // Cleanup
        pipeline.unload()

        print("\n" + String(repeating: "=", count: 60))
    }

    private func parseModel(_ id: String) -> VoxtralPipeline.Model? {
        switch id.lowercased() {
        case "mini-3b", "mini3b": return .mini3b
        case "mini-3b-8bit", "mini3b8bit": return .mini3b8bit
        case "mini-3b-4bit", "mini3b4bit": return .mini3b4bit
        case "small-24b", "small24b": return .small24b
        case "small-24b-8bit", "small24b8bit": return .small24b8bit
        case "small-4bit", "small4bit": return .small4bit
        default: return nil
        }
    }
}

// MARK: - Chat Command (Uses VoxtralPipeline ONLY)

struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "Ask a question about an audio file using VoxtralPipeline API"
    )

    @Argument(help: "Path to audio file")
    var audioFile: String

    @Argument(help: "Question or prompt about the audio")
    var prompt: String

    @Option(name: .shortAndLong, help: "Model: mini-3b, mini-3b-8bit, mini-3b-4bit, small-24b, small-24b-8bit, small-4bit")
    var model: String = "mini-3b-8bit"

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 500

    @Option(name: [.customShort("t"), .long], help: "Temperature for generation (0.0 = deterministic)")
    var temperature: Float = 0.7

    @Option(name: .shortAndLong, help: "Backend: 'mlx' (GPU only) or 'hybrid' (Core ML encoder + MLX decoder)")
    var backend: String = "mlx"

    func run() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VOXTRAL CHAT (VoxtralPipeline API)")
        print(String(repeating: "=", count: 60))

        // Validate audio file
        let audioURL = URL(fileURLWithPath: audioFile)
        guard FileManager.default.fileExists(atPath: audioFile) else {
            throw ValidationError("Audio file not found: \(audioFile)")
        }

        // Parse model option
        guard let pipelineModel = parseModel(model) else {
            throw ValidationError("Unknown model: \(model). Use 'voxtral list' to see available models.")
        }

        // Parse backend option
        let pipelineBackend: VoxtralPipeline.Backend = backend.lowercased() == "hybrid" ? .hybrid : .mlx
        print("\nModel: \(pipelineModel.displayName)")
        print("Backend: \(pipelineBackend.displayName)")
        print("Prompt: \(prompt)")

        // Create pipeline with high-level API
        var config = VoxtralPipeline.Configuration.default
        config.maxTokens = maxTokens
        config.temperature = temperature

        let pipeline = VoxtralPipeline(
            model: pipelineModel,
            backend: pipelineBackend,
            configuration: config
        )

        // Load model
        print("\n[1/3] Loading model...")
        let startLoad = Date()
        try await pipeline.loadModel { progress, status in
            print("  [\(Int(progress * 100))%] \(status)")
        }
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Model loaded in \(String(format: "%.2f", loadTime))s")
        print("  Encoder: \(pipeline.encoderStatus)")

        // Chat
        print("\n[2/3] Processing audio and prompt...")
        print("\n[3/3] Generating response...")
        let startGen = Date()

        let response = try await pipeline.chat(audio: audioURL, prompt: prompt)

        let genTime = Date().timeIntervalSince(startGen)

        // Output result
        print("\n" + String(repeating: "-", count: 60))
        print(response)
        print(String(repeating: "-", count: 60))

        // Statistics
        print("\nGenerated in \(String(format: "%.2f", genTime))s")
        print("Memory: \(pipeline.memorySummary)")

        // Cleanup
        pipeline.unload()

        print("\n" + String(repeating: "=", count: 60))
    }

    private func parseModel(_ id: String) -> VoxtralPipeline.Model? {
        switch id.lowercased() {
        case "mini-3b", "mini3b": return .mini3b
        case "mini-3b-8bit", "mini3b8bit": return .mini3b8bit
        case "mini-3b-4bit", "mini3b4bit": return .mini3b4bit
        case "small-24b", "small24b": return .small24b
        case "small-24b-8bit", "small24b8bit": return .small24b8bit
        case "small-4bit", "small4bit": return .small4bit
        default: return nil
        }
    }
}
