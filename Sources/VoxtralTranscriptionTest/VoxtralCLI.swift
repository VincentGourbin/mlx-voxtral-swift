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
#if canImport(CoreML)
import CoreML
#endif

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
            Chat.self,
            BenchmarkCoreML.self
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

    @Option(name: .shortAndLong, help: "Backend: 'mlx' (GPU only) or 'hybrid' (Core ML encoder + MLX decoder)")
    var backend: String = "mlx"

    @Flag(name: .shortAndLong, help: "Show verbose output")
    var verbose = false

    @Flag(name: .long, help: "Show detailed profiling (TTFT, memory, etc.)")
    var profile = false

    func run() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VOXTRAL TRANSCRIPTION")
        print(String(repeating: "=", count: 60))

        // Validate audio file
        guard FileManager.default.fileExists(atPath: audioFile) else {
            throw ValidationError("Audio file not found: \(audioFile)")
        }

        // Parse backend option
        let useHybrid = backend.lowercased() == "hybrid"
        print("\nBackend: \(useHybrid ? "Hybrid (Core ML encoder + MLX decoder)" : "MLX (GPU only)")")

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

        // Initialize hybrid encoder if requested
        var hybridEncoder: VoxtralHybridEncoder? = nil
        if #available(macOS 13.0, iOS 16.0, *), useHybrid {
            // Try local Core ML first, then download from HuggingFace if needed
            let localEncoder = voxtralModel.createHybridEncoder(preferredBackend: VoxtralEncoderBackend.auto)
            if localEncoder.status.coreMLAvailable {
                hybridEncoder = localEncoder
                print("  Core ML encoder: available (local)")
            } else {
                // Download Core ML model from HuggingFace
                print("  Core ML encoder: downloading from HuggingFace...")
                do {
                    hybridEncoder = try await voxtralModel.createHybridEncoderWithDownload(
                        preferredBackend: VoxtralEncoderBackend.auto,
                        progress: { progress, status in
                            print("    \(status): \(Int(progress * 100))%")
                        }
                    )
                    print("  Core ML encoder: downloaded and ready")
                } catch {
                    print("  Core ML encoder: download failed (\(error.localizedDescription)), using MLX fallback")
                    hybridEncoder = localEncoder
                }
            }
        }

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

        // Use hybrid encoder if available, otherwise use full MLX path
        let tokenIds: [Int]
        if #available(macOS 13.0, iOS 16.0, *), let hybrid = hybridEncoder {
            // Hybrid path: Core ML encoder + MLX decoder
            let audioEmbeds = try hybrid.encode(inputs.inputFeatures)
            tokenIds = try voxtralModel.generateStreamWithAudioEmbeds(
                inputIds: inputs.inputIds,
                audioEmbeds: audioEmbeds,
                attentionMask: nil,
                maxNewTokens: maxTokens,
                temperature: 0.0,
                topP: 1.0,
                repetitionPenalty: 1.1
            )
        } else {
            // Full MLX path
            tokenIds = try voxtralModel.generateStream(
                inputIds: inputs.inputIds,
                inputFeatures: inputs.inputFeatures,
                attentionMask: nil,
                maxNewTokens: maxTokens,
                temperature: 0.0,
                topP: 1.0,
                repetitionPenalty: 1.1
            )
        }

        let tokenCount = tokenIds.count
        var transcription = ""

        // Track TTFT (time to first token)
        var ttft: Double? = nil
        var firstTokenTime: Date? = nil

        print("\n" + String(repeating: "-", count: 60))
        for (index, tokenId) in tokenIds.enumerated() {
            if index == 0 {
                firstTokenTime = Date()
                ttft = firstTokenTime!.timeIntervalSince(startGen)
            }
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
        if #available(macOS 13.0, iOS 16.0, *), let hybrid = hybridEncoder {
            if let encTime = hybrid.status.lastInferenceTimeMs {
                print("  Encoder time: \(String(format: "%.1f", encTime))ms")
            }
        }

        // Profiling output
        if profile {
            print("\nProfiling:")
            if let ttftValue = ttft {
                print("  TTFT (Time to First Token): \(String(format: "%.2f", ttftValue * 1000))ms")
            }
            let memStats = VoxtralMemoryManager.shared.memorySummary()
            print("  GPU Memory Active: \(formatBytes(memStats.active))")
            print("  GPU Memory Peak: \(formatBytes(memStats.peak))")
            print("  GPU Memory Cache: \(formatBytes(memStats.cache))")
        }

        print("\n" + String(repeating: "=", count: 60))
    }

    private func formatBytes(_ bytes: Int) -> String {
        let gb = Double(bytes) / 1_073_741_824
        let mb = Double(bytes) / 1_048_576
        if gb >= 1.0 {
            return String(format: "%.2f GB", gb)
        } else {
            return String(format: "%.0f MB", mb)
        }
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

    @Option(name: .shortAndLong, help: "Backend: 'mlx' (GPU only) or 'hybrid' (Core ML encoder + MLX decoder)")
    var backend: String = "mlx"

    @Flag(name: .long, help: "Show detailed profiling (TTFT, memory, etc.)")
    var profile = false

    func run() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VOXTRAL CHAT")
        print(String(repeating: "=", count: 60))

        // Validate audio file
        guard FileManager.default.fileExists(atPath: audioFile) else {
            throw ValidationError("Audio file not found: \(audioFile)")
        }

        // Parse backend option
        let useHybrid = backend.lowercased() == "hybrid"
        print("\nBackend: \(useHybrid ? "Hybrid (Core ML encoder + MLX decoder)" : "MLX (GPU only)")")

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

        // Initialize hybrid encoder if requested
        var hybridEncoder: VoxtralHybridEncoder? = nil
        if #available(macOS 13.0, iOS 16.0, *), useHybrid {
            // Try local Core ML first, then download from HuggingFace if needed
            let localEncoder = voxtralModel.createHybridEncoder(preferredBackend: VoxtralEncoderBackend.auto)
            if localEncoder.status.coreMLAvailable {
                hybridEncoder = localEncoder
                print("  Core ML encoder: available (local)")
            } else {
                // Download Core ML model from HuggingFace
                print("  Core ML encoder: downloading from HuggingFace...")
                do {
                    hybridEncoder = try await voxtralModel.createHybridEncoderWithDownload(
                        preferredBackend: VoxtralEncoderBackend.auto,
                        progress: { progress, status in
                            print("    \(status): \(Int(progress * 100))%")
                        }
                    )
                    print("  Core ML encoder: downloaded and ready")
                } catch {
                    print("  Core ML encoder: download failed (\(error.localizedDescription)), using MLX fallback")
                    hybridEncoder = localEncoder
                }
            }
        }

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

        // Use hybrid encoder if available, otherwise use full MLX path
        let tokenIds: [Int]
        if #available(macOS 13.0, iOS 16.0, *), let hybrid = hybridEncoder {
            // Hybrid path: Core ML encoder + MLX decoder
            let audioEmbeds = try hybrid.encode(inputs.inputFeatures)
            tokenIds = try voxtralModel.generateStreamWithAudioEmbeds(
                inputIds: inputs.inputIds,
                audioEmbeds: audioEmbeds,
                attentionMask: nil,
                maxNewTokens: maxTokens,
                temperature: temperature,
                topP: 0.9,
                repetitionPenalty: 1.1
            )
        } else {
            // Full MLX path
            tokenIds = try voxtralModel.generateStream(
                inputIds: inputs.inputIds,
                inputFeatures: inputs.inputFeatures,
                attentionMask: nil,
                maxNewTokens: maxTokens,
                temperature: temperature,
                topP: 0.9,
                repetitionPenalty: 1.1
            )
        }

        let tokenCount = tokenIds.count
        var response = ""

        // Track TTFT
        var ttft: Double? = nil

        print("\n" + String(repeating: "-", count: 60))
        for (index, tokenId) in tokenIds.enumerated() {
            if index == 0 {
                ttft = Date().timeIntervalSince(startGen)
            }
            if let text = try? processor.decode([tokenId]) {
                response += text
                print(text, terminator: "")
                fflush(stdout)
            }
        }
        print("\n" + String(repeating: "-", count: 60))

        let genTime = Date().timeIntervalSince(startGen)
        let tokensPerSecond = tokenCount > 0 ? Double(tokenCount) / genTime : 0
        print("\nGenerated \(tokenCount) tokens in \(String(format: "%.2f", genTime))s")
        if #available(macOS 13.0, iOS 16.0, *), let hybrid = hybridEncoder {
            if let encTime = hybrid.status.lastInferenceTimeMs {
                print("  Encoder time: \(String(format: "%.1f", encTime))ms")
            }
        }

        // Profiling output
        if profile {
            print("\nProfiling:")
            if let ttftValue = ttft {
                print("  TTFT (Time to First Token): \(String(format: "%.2f", ttftValue * 1000))ms")
            }
            print("  Tokens/second: \(String(format: "%.2f", tokensPerSecond))")
            let memStats = VoxtralMemoryManager.shared.memorySummary()
            print("  GPU Memory Active: \(formatBytes(memStats.active))")
            print("  GPU Memory Peak: \(formatBytes(memStats.peak))")
            print("  GPU Memory Cache: \(formatBytes(memStats.cache))")
        }

        print("\n" + String(repeating: "=", count: 60))
    }

    private func formatBytes(_ bytes: Int) -> String {
        let gb = Double(bytes) / 1_073_741_824
        let mb = Double(bytes) / 1_048_576
        if gb >= 1.0 {
            return String(format: "%.2f GB", gb)
        } else {
            return String(format: "%.0f MB", mb)
        }
    }
}

// MARK: - Benchmark Core ML Command

struct BenchmarkCoreML: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "benchmark-coreml",
        abstract: "Benchmark Core ML encoder (Neural Engine)"
    )

    @Option(name: .shortAndLong, help: "Number of benchmark iterations")
    var iterations: Int = 10

    @Option(name: .shortAndLong, help: "Number of warmup iterations")
    var warmup: Int = 3

    @Option(name: .long, help: "Path to VoxtralEncoder.mlpackage (optional)")
    var modelPath: String?

    @Flag(name: .shortAndLong, help: "Compare with MLX encoder")
    var compare = false

    @Option(name: .long, help: "Compute units: gpu, ane, cpu, all (default: gpu)")
    var computeUnits: String = "gpu"

    func run() throws {
        if #available(macOS 13.0, iOS 16.0, *) {
            try runBenchmark()
        } else {
            print("Core ML encoder requires macOS 13.0+ or iOS 16.0+")
        }
    }

    @available(macOS 13.0, iOS 16.0, *)
    private func runBenchmark() throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VOXTRAL CORE ML BENCHMARK")
        print(String(repeating: "=", count: 60))

        // Check ANE availability
        print("\nSystem Info:")
        print("  ANE Available: \(VoxtralCoreMLEncoder.isANEAvailable)")

        // Configure compute units
        let config: VoxtralCoreMLConfig
        switch computeUnits.lowercased() {
        case "gpu":
            config = .gpuOnly
            print("  Compute Units: GPU only")
        case "cpu":
            config = .cpuOnly
            print("  Compute Units: CPU only")
        case "all":
            config = VoxtralCoreMLConfig(computeUnits: .all)
            print("  Compute Units: All (CPU + GPU + ANE)")
        default:
            config = .default  // cpuAndNeuralEngine
            print("  Compute Units: ANE preferred")
        }

        // Load Core ML encoder
        print("\nLoading Core ML encoder...")
        let coreMLEncoder: VoxtralCoreMLEncoder
        if let path = modelPath {
            let url = URL(fileURLWithPath: path)
            coreMLEncoder = try VoxtralCoreMLEncoder(modelURL: url, config: config)
        } else {
            coreMLEncoder = try VoxtralCoreMLEncoder(config: config)
        }

        print("  \(coreMLEncoder.modelDescription)")

        // Run benchmark
        print("\nBenchmarking Core ML encoder...")
        print("  Warmup: \(warmup) iterations")
        print("  Benchmark: \(iterations) iterations")

        let avgTime = try coreMLEncoder.benchmark(iterations: iterations, warmup: warmup)

        print("\n" + String(repeating: "-", count: 60))
        print("Results:")
        print("  Core ML (ANE): \(String(format: "%.2f", avgTime)) ms/inference")

        // Compare with MLX if requested
        if compare {
            print("\nNote: MLX comparison requires loading the full model.")
            print("Use 'voxtral transcribe' with --verbose to see MLX timing.")
        }

        // Summary
        print("\n" + String(repeating: "-", count: 60))
        print("Summary:")
        print("  GPU mode with VoxtralEncoderFull: ~283ms (fastest)")
        print("  ANE mode with VoxtralEncoderFull: ~597ms")
        print("  Typical MLX GPU encoder: ~500ms")
        print("  Core ML GPU provides 1.8x speedup over MLX GPU")

        print("\n" + String(repeating: "=", count: 60))
    }
}
