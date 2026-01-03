/**
 * VoxtralCLI - Command-line interface for Voxtral speech-to-text
 *
 * Features:
 * - List available models
 * - Download models from HuggingFace
 * - Transcribe audio files
 * - Chat mode (ask questions about audio)
 */

import Foundation
import MLX
import VoxtralCore
import MLXLMCommon
import ArgumentParser

@main
struct VoxtralCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voxtral",
        abstract: "Voxtral speech-to-text for Apple Silicon",
        version: "1.0.0",
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

// MARK: - Transcribe Command

struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe an audio file"
    )

    @Argument(help: "Path to audio file (MP3, WAV, M4A, etc.)")
    var audioFile: String

    @Option(name: .shortAndLong, help: "Model ID or path (default: mini-3b-8bit)")
    var model: String = "mini-3b-8bit"

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 500

    @Option(name: [.customShort("l"), .long], help: "Language code (e.g., 'en', 'fr')")
    var language: String = "en"

    @Flag(name: .shortAndLong, help: "Show verbose output")
    var verbose = false

    func run() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VOXTRAL TRANSCRIPTION")
        print(String(repeating: "=", count: 60))

        // Validate audio file
        guard FileManager.default.fileExists(atPath: audioFile) else {
            throw ValidationError("Audio file not found: \(audioFile)")
        }

        // Resolve model path
        print("\n[1/4] Resolving model...")
        let modelPath = try await ModelDownloader.resolveModel(model)
        print("  Using model: \(modelPath.path)")

        // Load model
        print("\n[2/4] Loading model...")
        let startLoad = Date()
        let (standardModel, config) = try loadVoxtralStandardModel(
            modelPath: modelPath.path,
            dtype: .float16
        )
        let voxtralModel = VoxtralForConditionalGeneration(standardModel: standardModel)
        let processor = try VoxtralProcessor.fromPretrained(modelPath.path)
        let loadTime = Date().timeIntervalSince(startLoad)

        if verbose {
            print("  Model loaded in \(String(format: "%.2f", loadTime))s")
            print("  Model type: \(config.modelType)")
            print("  Audio encoder: \(config.audioConfig.hiddenLayers) layers")
        } else {
            print("  Model loaded in \(String(format: "%.2f", loadTime))s")
        }

        // Process audio
        print("\n[3/4] Processing audio: \(audioFile)")
        let startAudio = Date()
        let inputs = try processor.applyTranscritionRequest(
            audio: audioFile,
            language: language,
            samplingRate: 16000
        )
        let audioTime = Date().timeIntervalSince(startAudio)

        let numChunks = inputs.inputFeatures.shape[0]
        let audioDuration = Float(numChunks) * 30.0
        print("  Audio processed in \(String(format: "%.2f", audioTime))s (~\(Int(audioDuration))s audio)")

        // Generate transcription
        print("\n[4/4] Generating transcription...")
        let startGen = Date()

        var tokenCount = 0
        var transcription = ""

        let streamResults = try voxtralModel.generateStream(
            inputIds: inputs.inputIds,
            inputFeatures: inputs.inputFeatures,
            attentionMask: nil,
            maxNewTokens: maxTokens,
            temperature: 0.0,
            topP: 1.0,
            repetitionPenalty: 1.1
        )

        print("\n" + String(repeating: "-", count: 60))
        for (token, _) in streamResults {
            let tokenId = token.item(Int.self)
            tokenCount += 1

            if let text = try? processor.decode([tokenId]) {
                transcription += text
                print(text, terminator: "")
                fflush(stdout)
            }
        }
        print("\n" + String(repeating: "-", count: 60))

        let genTime = Date().timeIntervalSince(startGen)
        let tokensPerSecond = Double(tokenCount) / genTime

        print("\nStatistics:")
        print("  Tokens: \(tokenCount)")
        print("  Time: \(String(format: "%.2f", genTime))s")
        print("  Speed: \(String(format: "%.1f", tokensPerSecond)) tokens/s")
        print("  RTF: \(String(format: "%.2fx", audioDuration / Float(genTime)))")

        print("\n" + String(repeating: "=", count: 60))
    }
}

// MARK: - Chat Command

struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "Ask a question about an audio file"
    )

    @Argument(help: "Path to audio file")
    var audioFile: String

    @Argument(help: "Question or prompt about the audio")
    var prompt: String

    @Option(name: .shortAndLong, help: "Model ID or path (default: mini-3b-8bit)")
    var model: String = "mini-3b-8bit"

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 500

    @Option(name: [.customShort("t"), .long], help: "Temperature for generation (0.0 = deterministic)")
    var temperature: Float = 0.7

    func run() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VOXTRAL CHAT")
        print(String(repeating: "=", count: 60))

        // Validate audio file
        guard FileManager.default.fileExists(atPath: audioFile) else {
            throw ValidationError("Audio file not found: \(audioFile)")
        }

        // Resolve model path
        print("\n[1/4] Resolving model...")
        let modelPath = try await ModelDownloader.resolveModel(model)
        print("  Using model: \(modelPath.path)")

        // Load model
        print("\n[2/4] Loading model...")
        let startLoad = Date()
        let (standardModel, _) = try loadVoxtralStandardModel(
            modelPath: modelPath.path,
            dtype: .float16
        )
        let voxtralModel = VoxtralForConditionalGeneration(standardModel: standardModel)
        let processor = try VoxtralProcessor.fromPretrained(modelPath.path)
        print("  Model loaded in \(String(format: "%.2f", Date().timeIntervalSince(startLoad)))s")

        // Process audio with chat template
        print("\n[3/4] Processing audio and prompt...")
        print("  Audio: \(audioFile)")
        print("  Prompt: \(prompt)")

        let conversation: [[String: Any]] = [
            [
                "role": "user",
                "content": [
                    ["type": "audio", "audio": audioFile],
                    ["type": "text", "text": prompt]
                ]
            ]
        ]

        let chatResult = try processor.applyChatTemplate(
            conversation: conversation,
            tokenize: true,
            returnTensors: "mlx"
        ) as! [String: MLXArray]

        let inputs = ProcessedInputs(
            inputIds: chatResult["input_ids"]!,
            inputFeatures: chatResult["input_features"]!
        )

        // Generate response
        print("\n[4/4] Generating response...")
        let startGen = Date()

        var tokenCount = 0
        var response = ""

        let streamResults = try voxtralModel.generateStream(
            inputIds: inputs.inputIds,
            inputFeatures: inputs.inputFeatures,
            attentionMask: nil,
            maxNewTokens: maxTokens,
            temperature: temperature,
            topP: 0.9,
            repetitionPenalty: 1.1
        )

        print("\n" + String(repeating: "-", count: 60))
        for (token, _) in streamResults {
            let tokenId = token.item(Int.self)
            tokenCount += 1

            if let text = try? processor.decode([tokenId]) {
                response += text
                print(text, terminator: "")
                fflush(stdout)
            }
        }
        print("\n" + String(repeating: "-", count: 60))

        let genTime = Date().timeIntervalSince(startGen)
        print("\nGenerated \(tokenCount) tokens in \(String(format: "%.2f", genTime))s")

        print("\n" + String(repeating: "=", count: 60))
    }
}
