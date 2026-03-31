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
        abstract: "Voxtral speech-to-text & text-to-speech for Apple Silicon",
        version: "2.0.0",
        subcommands: [
            ListModels.self,
            Download.self,
            Transcribe.self,
            Chat.self,
            TTS.self,
            Realtime.self
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
            let hasSttModels = !downloadedModels.isEmpty
            let hasTtsModels = VoxtralTTSRegistry.models.contains { ModelDownloader.isTTSModelDownloaded($0) }

            if !hasSttModels && !hasTtsModels {
                print("\nNo models downloaded yet.")
                print("Use 'voxtral download <model-id>' to download a model.")
                print("Use 'voxtral list' to see available models.")
            } else {
                print("\n" + String(repeating: "=", count: 60))
                print("DOWNLOADED MODELS")
                print(String(repeating: "=", count: 60))
                for model in downloadedModels {
                    if let path = ModelDownloader.findModelPath(for: model) {
                        print("\n  [STT] \(model.id): \(model.name)")
                        print("    Path: \(path.path)")
                    }
                }
                for model in VoxtralTTSRegistry.models {
                    if let path = ModelDownloader.findTTSModelPath(for: model) {
                        print("\n  [TTS] \(model.id): \(model.name)")
                        print("    Path: \(path.path)")
                    }
                }
                for model in VoxtralRealtimeRegistry.models {
                    if let path = ModelDownloader.findRealtimeModelPath(for: model) {
                        print("\n  [REALTIME] \(model.id): \(model.name)")
                        print("    Path: \(path.path)")
                    }
                }
            }
        } else {
            // STT models
            ModelRegistry.printAvailableModels()
            print("\nSTT download status:")
            for model in ModelRegistry.models {
                let status = ModelDownloader.findModelPath(for: model) != nil ? "[downloaded]" : "[not downloaded]"
                print("  \(model.id): \(status)")
            }

            // TTS models
            print("\n" + String(repeating: "-", count: 40))
            VoxtralTTSRegistry.printAvailableModels()
            print("\nTTS download status:")
            for model in VoxtralTTSRegistry.models {
                let status = ModelDownloader.findTTSModelPath(for: model) != nil ? "[downloaded]" : "[not downloaded]"
                print("  \(model.id): \(status)")
            }

            // Realtime models
            print("\n" + String(repeating: "-", count: 40))
            VoxtralRealtimeRegistry.printAvailableModels()
            print("\nRealtime download status:")
            for model in VoxtralRealtimeRegistry.models {
                let status = ModelDownloader.findRealtimeModelPath(for: model) != nil ? "[downloaded]" : "[not downloaded]"
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

        // Check if it's a Realtime model
        if let realtimeModel = VoxtralRealtimeRegistry.model(withId: model) {
            let modelPath = try await ModelDownloader.downloadRealtimeModel(realtimeModel) { progress, message in
                print("[\(Int(progress * 100))%] \(message)")
            }
            print("\n" + String(repeating: "=", count: 60))
            print("Download complete!")
            print("Model path: \(modelPath.path)")
            print("\nTo transcribe audio:")
            print("  voxtral realtime <audio-file>")
            print(String(repeating: "=", count: 60))
        // Check if it's a TTS model
        } else if let ttsModel = VoxtralTTSRegistry.model(withId: model) {
            let modelPath = try await ModelDownloader.downloadTTSModel(ttsModel) { progress, message in
                print("[\(Int(progress * 100))%] \(message)")
            }
            print("\n" + String(repeating: "=", count: 60))
            print("Download complete!")
            print("Model path: \(modelPath.path)")
            print("\nTo synthesize speech:")
            print("  voxtral tts \"Hello world\" -o output.wav")
            print(String(repeating: "=", count: 60))
        } else {
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

// MARK: - TTS Command (Uses VoxtralTTSPipeline)

@available(macOS 14.0, *)
struct TTS: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tts",
        abstract: "Convert text to speech using Voxtral TTS"
    )

    @Argument(help: "Text to convert to speech")
    var text: String

    @Option(name: .shortAndLong, help: "Output WAV file path")
    var output: String = "output.wav"

    @Option(name: .shortAndLong, help: "Voice preset (use 'voxtral list' to see available voices)")
    var voice: String = "neutral_female"

    @Option(name: .long, help: "ZeroVoice 3D coordinate 'x,y,z' for procedural voice (overrides --voice)")
    var voiceXyz: String?

    @Option(name: .long, help: "Blend two voices 'voiceA+voiceB:weight' (e.g. 'neutral_female+fr_male:0.15')")
    var blend: String?

    @Option(name: .long, help: "Maximum audio frames to generate (12.5 frames/sec)")
    var maxFrames: Int = 2500

    @Option(name: [.customShort("a"), .long], help: "CFG alpha for flow matching (default: 1.2)")
    var cfgAlpha: Float = 1.2

    @Option(name: .long, help: "Number of Euler steps for flow matching (default: 8)")
    var flowSteps: Int = 8

    @Option(name: [.customShort("t"), .long], help: "Temperature for semantic token sampling (0 = greedy)")
    var temperature: Float = 0.0

    func run() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VOXTRAL TTS (Text-to-Speech)")
        print(String(repeating: "=", count: 60))

        print("\nText: \(text)")
        print("Output: \(output)")

        // Create pipeline
        var config = VoxtralTTSPipeline.Configuration.default
        config.maxFrames = maxFrames
        config.cfgAlpha = cfgAlpha
        config.flowSteps = flowSteps
        config.temperature = temperature

        let pipeline = VoxtralTTSPipeline(configuration: config)

        // Load model
        print("\n[1/3] Loading TTS model...")
        let startLoad = Date()
        try await pipeline.loadModel { progress, status in
            print("  [\(Int(progress * 100))%] \(status)")
        }
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Model loaded in \(String(format: "%.2f", loadTime))s")

        // Synthesize with appropriate voice mode
        print("\n[2/3] Generating speech...")
        let result: TTSSynthesisResult

        if let xyz = voiceXyz {
            // ZeroVoice coordinate mode
            let parts = xyz.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard parts.count == 3 else {
                throw ValidationError("--voice-xyz must be 'x,y,z' (e.g. '50,50,50')")
            }
            let coord = (x: parts[0], y: parts[1], z: parts[2])
            if let recipe = pipeline.voiceRecipe(x: coord.x, y: coord.y, z: coord.z) {
                print("  ZeroVoice: \(recipe.voiceA.rawValue) + \(recipe.voiceB.rawValue) @ \(String(format: "%.2f", recipe.blendWeight))")
            }
            result = try await pipeline.synthesize(text: text, voiceCoordinate: coord)

        } else if let blendSpec = blend {
            // Manual blend mode: "voiceA+voiceB:weight"
            let mainParts = blendSpec.split(separator: ":")
            guard mainParts.count == 2,
                  let weight = Float(mainParts[1]) else {
                throw ValidationError("--blend must be 'voiceA+voiceB:weight' (e.g. 'neutral_female+fr_male:0.15')")
            }
            let voiceParts = mainParts[0].split(separator: "+")
            guard voiceParts.count == 2,
                  let voiceA = VoxtralVoice(rawValue: String(voiceParts[0])),
                  let voiceB = VoxtralVoice(rawValue: String(voiceParts[1])) else {
                throw ValidationError("Unknown voice names in blend spec. Use 'voxtral list' to see available voices.")
            }
            guard let blended = pipeline.blendVoicePresets(voiceA, voiceB, t: weight) else {
                throw ValidationError("Voice embeddings not loaded for blend")
            }
            print("  Blend: \(voiceA.rawValue) + \(voiceB.rawValue) @ \(String(format: "%.2f", weight))")
            result = try await pipeline.synthesize(text: text, voiceEmbedding: blended)

        } else {
            // Standard preset voice mode
            guard let voicePreset = VoxtralVoice(rawValue: voice) else {
                print("Unknown voice: \(voice)")
                print("\nAvailable voices:")
                for v in VoxtralVoice.allCases { print("  \(v.rawValue)") }
                throw ValidationError("Unknown voice: \(voice)")
            }
            print("  Voice: \(voicePreset.displayName)")
            result = try await pipeline.synthesize(text: text, voice: voicePreset)
        }

        // Save WAV
        print("\n[3/3] Saving audio...")
        let outputURL = URL(fileURLWithPath: output)
        try WAVWriter.write(waveform: result.waveform, to: outputURL)

        // Statistics
        print("\n" + String(repeating: "-", count: 60))
        print("Audio saved to: \(output)")
        print(String(repeating: "-", count: 60))
        print("\nStatistics:")
        print("  Duration: \(String(format: "%.2f", result.duration))s")
        print("  Frames: \(result.numFrames)")
        print("  Generation time: \(String(format: "%.2f", result.generationTime))s")
        print("  Real-time factor: \(String(format: "%.2f", result.realTimeFactor))x")
        print("  Frames/sec: \(String(format: "%.1f", result.framesPerSecond))")

        // Cleanup
        pipeline.unload()

        print("\n" + String(repeating: "=", count: 60))
    }
}

// MARK: - Realtime Command (Uses VoxtralRealtimePipeline)

@available(macOS 14.0, *)
struct Realtime: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "realtime",
        abstract: "Transcribe using Voxtral Realtime 4B (streaming model)"
    )

    @Argument(help: "Path to audio file (MP3, WAV, M4A, etc.)")
    var audioFile: String

    @Option(name: .shortAndLong, help: "Model: realtime-4b-4bit, realtime-4b-fp16, realtime-4b")
    var model: String = "realtime-4b-4bit"

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 4096

    @Option(name: .long, help: "Transcription delay in ms (default: 480)")
    var delay: Int = 480

    @Option(name: [.customShort("t"), .long], help: "Temperature (0.0 = greedy)")
    var temperature: Float = 0.0

    @Flag(name: .long, help: "Extract and print audio embeddings shape")
    var embeddings = false

    func run() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VOXTRAL REALTIME (Streaming STT)")
        print(String(repeating: "=", count: 60))

        guard FileManager.default.fileExists(atPath: audioFile) else {
            throw ValidationError("Audio file not found: \(audioFile)")
        }

        print("\nModel: \(model)")
        print("Delay: \(delay)ms")

        var config = VoxtralRealtimePipeline.Configuration.default
        config.maxTokens = maxTokens
        config.temperature = temperature
        config.transcriptionDelayMs = delay

        let pipeline = VoxtralRealtimePipeline(configuration: config)

        // Load model
        print("\n[1/3] Loading Realtime model...")
        let startLoad = Date()
        try await pipeline.loadModel(modelId: model) { progress, status in
            print("  [\(Int(progress * 100))%] \(status)")
        }
        let loadTime = Date().timeIntervalSince(startLoad)
        print("  Model loaded in \(String(format: "%.2f", loadTime))s")

        // Extract embeddings if requested
        if embeddings {
            print("\n[2/3] Extracting audio embeddings...")
            let audioURL = URL(fileURLWithPath: audioFile)
            let embeds = try await pipeline.extractAudioEmbeddings(audio: audioURL)
            print("  Embeddings shape: \(embeds.shape)")
            print("  dtype: \(embeds.dtype)")
        }

        // Transcribe
        print("\n[\(embeddings ? "3" : "2")/3] Transcribing...")
        let startGen = Date()
        let audioURL = URL(fileURLWithPath: audioFile)
        let transcription = try await pipeline.transcribe(audio: audioURL)
        let genTime = Date().timeIntervalSince(startGen)

        print("\n" + String(repeating: "-", count: 60))
        print(transcription)
        print(String(repeating: "-", count: 60))

        print("\nStatistics:")
        print("  Time: \(String(format: "%.2f", genTime))s")

        pipeline.unload()
        print("\n" + String(repeating: "=", count: 60))
    }
}
