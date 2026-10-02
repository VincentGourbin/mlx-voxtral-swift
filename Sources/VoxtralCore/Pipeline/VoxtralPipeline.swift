/**
 * VoxtralPipeline - Simplified facade API for Voxtral transcription
 *
 * Provides a unified, easy-to-use interface aligned with flux-2-swift-mlx patterns.
 * Supports both pure MLX and hybrid Core ML + MLX modes.
 *
 * Usage:
 * ```swift
 * let pipeline = VoxtralPipeline(model: .mini3b8bit)
 * try await pipeline.loadModel()
 * let text = try await pipeline.transcribe(audio: audioURL)
 * pipeline.unload()
 * ```
 */

import AVFoundation
import Foundation
import MLX
import MLXNN
import MLXProfiler

/// Simplified facade for Voxtral speech-to-text
public class VoxtralPipeline: @unchecked Sendable {

    // MARK: - Model Selection

    /// Available Voxtral model variants
    public enum Model: String, CaseIterable, Sendable {
        case mini3b = "mini-3b"
        case mini3b8bit = "mini-3b-8bit"
        case mini3b4bit = "mini-3b-4bit"
        case small24b = "small-24b"
        case small24b8bit = "small-24b-8bit"
        case small4bit = "small-4bit"

        /// HuggingFace repository of this model, read from `VoxtralModelRegistry`: one table for the pipeline, the app and
        /// the downloader (K-10, ASK-15)
        public var repoId: String {
            VoxtralModelRegistry.model(withId: rawValue)?.repoId ?? rawValue
        }

        /// Human-readable display name
        public var displayName: String {
            switch self {
            case .mini3b: return "Voxtral Mini 3B (Full)"
            case .mini3b8bit: return "Voxtral Mini 3B (8-bit)"
            case .mini3b4bit: return "Voxtral Mini 3B (4-bit)"
            case .small24b: return "Voxtral Small 24B (Full)"
            case .small24b8bit: return "Voxtral Small 24B (8-bit)"
            case .small4bit: return "Voxtral Small (4-bit)"
            }
        }

        /// Recommended model for most users
        public static var recommended: Model { .mini3b8bit }
    }

    // MARK: - Backend Selection

    /// Encoder backend for audio processing
    public enum Backend: Sendable {
        case mlx           // Pure MLX (GPU)
        case hybrid        // Core ML encoder + MLX decoder
        case auto          // Auto-detect best backend

        public var displayName: String {
            switch self {
            case .mlx: return "MLX (GPU)"
            case .hybrid: return "Hybrid (Core ML + MLX)"
            case .auto: return "Auto"
            }
        }
    }

    // MARK: - Configuration

    /// Pipeline configuration
    public struct Configuration: Sendable {
        /// Text-token budget. `nil` (default): proportional to the audio duration
        /// (`automaticMaxTokens(forDuration:)`), so a long audio is never cut silently; a value is a
        /// hard cap, reported by `lastResultTruncated` when reached (ASK-8 = A, K-5)
        public var maxTokens: Int?

        /// Sampling temperature (0 = deterministic)
        public var temperature: Float

        /// Nucleus sampling parameter
        public var topP: Float

        /// Repetition penalty
        public var repetitionPenalty: Float

        /// Memory optimization configuration
        public var memoryOptimization: MemoryOptimizationConfig

        /// Default configuration
        public static var `default`: Configuration {
            Configuration(
                maxTokens: nil,
                temperature: 0.0,
                topP: 0.95,
                repetitionPenalty: 1.2,
                memoryOptimization: .recommended()
            )
        }

        public init(
            maxTokens: Int? = nil,
            temperature: Float = 0.0,
            topP: Float = 0.95,
            repetitionPenalty: Float = 1.2,
            memoryOptimization: MemoryOptimizationConfig = .recommended()
        ) {
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
            self.repetitionPenalty = repetitionPenalty
            self.memoryOptimization = memoryOptimization
        }
    }

    // MARK: - Token budget (K-5)

    /// Speech rate the automatic budget allows: 1.5 × the densest measured rate (C-moyen FR:
    /// 693 tokens for 173.8 s ≈ 4.0 tokens/s; EN: 520 for 167 s ≈ 3.1), so a runaway loop still stops.
    static let automaticTokensPerSecond = 6.0

    /// Budget used when `maxTokens` is nil: ⌈duration × 6⌉ + 64, never below the former default 500.
    public static func automaticMaxTokens(forDuration seconds: Double) -> Int {
        max(500, Int((seconds * automaticTokensPerSecond).rounded(.up)) + 64)
    }

    /// True when the last `transcribe`/`chat` stopped on its token budget, not on an end token
    public private(set) var lastResultTruncated = false

    private func tokenBudget(for audio: URL) -> Int {
        if let explicit = configuration.maxTokens { return explicit }
        let seconds = (try? AVAudioFile(forReading: audio)).map { Double($0.length) / $0.processingFormat.sampleRate } ?? 0
        return Self.automaticMaxTokens(forDuration: seconds)
    }

    /// Tokens generated by the last `transcribe`/`chat` (K-27)
    public private(set) var lastTokenCount = 0

    private func recordTruncation(_ tokenIds: [Int], budget: Int) {
        lastTokenCount = tokenIds.count
        lastResultTruncated = tokenIds.count >= budget && !(tokenIds.last.map(voxtralModelStopTokens.contains) ?? false)
    }

    private var voxtralModelStopTokens: [Int] { voxtralModel?.stopTokenIds ?? [2, 4] }

    // MARK: - State

    /// Current pipeline state
    public enum State: Sendable {
        case unloaded
        case loading
        case ready
        case processing
        case error(String)

        /// Check if state matches unloaded
        var isUnloaded: Bool {
            if case .unloaded = self { return true }
            return false
        }

        /// Check if state matches ready
        var isReady: Bool {
            if case .ready = self { return true }
            return false
        }

        /// Check if state is an error state
        var isError: Bool {
            if case .error = self { return true }
            return false
        }
    }

    // MARK: - Properties

    /// Selected model variant
    public let model: Model

    /// Selected backend
    public let backend: Backend

    /// Current configuration
    public var configuration: Configuration

    /// Current pipeline state
    /// Read-only view of the gate: every transition happens under its lock (K-11)
    public var state: State { gate.state }

    /// State, running operation and generation token, changed atomically
    let gate = PipelineGate<State>(.unloaded)

    /// Opt-in MLX cache limit held while loaded (K-52)
    private let cachePolicy = MLXCachePolicy()

    /// Loaded Voxtral model
    private(set) var voxtralModel: VoxtralModel?

    /// Loaded processor
    private var processor: VoxtralProcessor?

    /// A local pack to load instead of `model`'s repository (tests of third-party packs, K-8)
    var modelDirectoryOverride: URL?

    /// Hybrid encoder (for hybrid mode)
    private var hybridEncoder: VoxtralHybridEncoder?

    /// Progress callback type
    public typealias ProgressCallback = @Sendable (Double, String) -> Void

    // MARK: - Initialization

    /// Create a new pipeline
    /// - Parameters:
    ///   - model: Model variant to use (default: .mini3b8bit)
    ///   - backend: Encoder backend (default: .auto)
    ///   - configuration: Generation configuration (default: .default)
    public init(
        model: Model = .recommended,
        backend: Backend = .auto,
        configuration: Configuration = .default
    ) {
        self.model = model
        self.backend = backend
        self.configuration = configuration

        // configuration.memoryOptimization is passed to each generation: the shared
        // VoxtralMemoryManager.config is never overwritten by a pipeline (S-11)
    }

    // MARK: - Model Loading

    /// Load the model
    /// - Parameter progress: Optional progress callback (progress 0-1, status message)
    public func loadModel(progress: ProgressCallback? = nil) async throws {
        // MLX errors become VoxtralError.mlx instead of terminating the host (K-1)
        try await withMLXErrors { _ in
            // Atomic check-and-set: a concurrent load or a running operation is refused (K-11)
            let generation = try gate.begin(
                "loading", accepts: { $0.isUnloaded || $0.isError },
                refusal: VoxtralPipelineError.invalidState("Model already loaded or loading"),
                busy: VoxtralPipelineError.busy, state: .loading, newGeneration: true)
            progress?(0.0, "Starting model download...")

            let beacon = VoxtralRuntimeBeacon.begin(task: "load-models", model: model.rawValue)
            defer { beacon?.end() }

            do {
                let profiler = MLXProfiler.shared
                let session = profiler.activeSession

                // Download/resolve model path
                progress?(0.1, "Downloading model...")
                session?.beginPhase("1. Model Download", category: .modelLoad)
                // By id: a local copy needs no network request (K-10)
                let modelPath: URL
                if let modelDirectoryOverride {
                    modelPath = modelDirectoryOverride
                } else {
                    modelPath = try await VoxtralModelDownloader.resolveModel(model.rawValue) { downloadProgress, status in
                        progress?(0.1 + downloadProgress * 0.4, status)
                    }
                }
                session?.endPhase("1. Model Download", category: .modelLoad)

                // Load model using the working loadVoxtralStandardModel approach
                progress?(0.5, "Loading model...")
                session?.beginPhase("2. Model Loading", category: .modelLoad)
                // Weights load off the cooperative pool (K-15); MLX errors caught on that queue (K-1)
                let (standardModel, _) = try await runOffCooperativePool {
                    try withMLXErrors { _ in try loadVoxtralStandardModel(modelPath: modelPath.path) }
                }
                self.voxtralModel = VoxtralForConditionalGeneration(standardModel: standardModel)
                session?.endPhase("2. Model Loading", category: .modelLoad)

                // Load tokenizer and setup encoder IN PARALLEL
                // CoreML compilation (encoder setup) can take 1-2 min on first run,
                // so we overlap it with tokenizer loading to reduce total wait time.
                progress?(0.6, "Loading tokenizer & compiling encoder...")

                session?.beginPhase("3. Tokenizer Loading", category: .tokenization)
                session?.beginPhase("4. Encoder Setup", category: .modelLoad)

                let capturedModelPath = modelPath.path
                async let tokenizerTask: VoxtralProcessor = {
                    try VoxtralProcessor.fromPretrained(capturedModelPath) { processorProgress, status in
                        progress?(0.6 + processorProgress * 0.15, status)
                    }
                }()
                async let encoderTask: Void = setupEncoder()

                self.processor = try await tokenizerTask
                // Stop on the tokenizer's own special tokens, never on a hard-coded id (K-4)
                if let tokenizer = self.processor?.tokenizer {
                    self.voxtralModel?.stopTokenIds = [tokenizer.eosToken, tokenizer.getControlToken("[/INST]")]
                }
                session?.endPhase("3. Tokenizer Loading", category: .tokenization)

                try await encoderTask
                session?.endPhase("4. Encoder Setup", category: .modelLoad)

                cachePolicy.apply(configuration.memoryOptimization.cacheLimitBytes)
                gate.end(generation, state: .ready)
                progress?(1.0, "Model ready!")

            } catch {
                gate.end(generation, state: .error(error.localizedDescription))
                throw error
            }
        }
    }

    /// Setup encoder based on backend preference
    /// Downloads Core ML model from HuggingFace if hybrid mode is requested
    private func setupEncoder() async throws {
        guard let voxtralModel = voxtralModel else {
            return
        }

        switch backend {
        case .hybrid, .auto:
            // Download Core ML model from HuggingFace (same cache as MLX models)
            do {
                hybridEncoder = try await voxtralModel.createHybridEncoderWithDownload(
                    preferredBackend: backend == .hybrid ? .coreML : .auto
                )
                VoxtralDebug.log("Hybrid encoder created with Core ML: \(hybridEncoder?.status.description ?? "nil")")
            } catch {
                // If Core ML download fails, fall back to pure MLX (said, not silent)
                VoxtralDebug.always("Core ML encoder unavailable (\(error.localizedDescription)); using the MLX encoder")
                hybridEncoder = voxtralModel.createHybridEncoder(
                    preferredBackend: .mlx
                )
            }
            if let status = hybridEncoder?.status {
                VoxtralDebug.always("Encoder: \(status.backend.displayName), Core ML available: \(status.coreMLAvailable)")
            }

        case .mlx:
            // Pure MLX mode - no hybrid encoder needed
            hybridEncoder = nil
        }
    }

    // MARK: - Transcription

    /// Transcribe audio file
    /// - Parameters:
    ///   - audio: URL to audio file
    ///   - language: Optional language code (e.g. `"fr"`, `"en"`). Pass `nil`
    ///     to let the model auto-detect the spoken language (the processor
    ///     omits the `lang:xx` prompt token in that case — cf.
    ///     `VoxtralProcessor.applyTranscriptionRequest`). Defaults to `nil`
    ///     so dubbing / multilingual workflows can rely on auto-detection
    ///     without hardcoding a source language.
    /// - Returns: Transcribed text
    public func transcribe(audio: URL, language: String? = nil) async throws -> String {
        // MLX errors become VoxtralError.mlx instead of terminating the host (K-1)
        // Off the cooperative pool, cancellable at every step (K-15)
        return try await runOffCooperativePool { [self] in try withMLXErrors { _ in
            let generation = try gate.begin(
                "transcription", accepts: { $0.isReady }, refusal: VoxtralPipelineError.invalidState("Model not loaded"),
                busy: VoxtralPipelineError.busy, state: .processing)
            defer {
                cachePolicy.endOfResponse()
                gate.end(generation, state: .ready)
            }

            guard let model = voxtralModel, let processor = processor else {
                throw VoxtralPipelineError.modelNotLoaded
            }

            let session = MLXProfiler.shared.activeSession
            let beacon = VoxtralRuntimeBeacon.begin(task: "transcribe", model: self.model.rawValue)
            defer {
                beacon?.end()
                // Apply memory optimization
                VoxtralMemoryManager.shared.optimizeIfNeeded(tokenIndex: 0, config: configuration.memoryOptimization)
            }

            // Create transcription request (note: method name has typo in original)
            session?.beginPhase("Audio Feature Extraction", category: .audioFeatureExtract)
            let inputs = try processor.applyTranscriptionRequest(
                audio: audio.path,
                language: language
            )
            session?.endPhase("Audio Feature Extraction", category: .audioFeatureExtract)

            // Generate transcription within the text-token budget (explicit, or from the duration: K-5)
            let budget = tokenBudget(for: audio)
            let tokenIds: [Int]

            if let hybrid = hybridEncoder, hybrid.status.coreMLAvailable {
                // Hybrid mode: use Core ML for audio encoding
                session?.beginPhase("Audio Encoding (CoreML)", category: .audioEncode)
                let audioEmbeds = try hybrid.encode(inputs.inputFeatures)
                session?.endPhase("Audio Encoding (CoreML)", category: .audioEncode)

                session?.beginPhase("Generation", category: .generation)
                tokenIds = try model.generateStreamWithAudioEmbeds(
                    inputIds: inputs.inputIds,
                    audioEmbeds: audioEmbeds,
                    maxNewTokens: budget,
                    temperature: configuration.temperature,
                    topP: configuration.topP,
                    repetitionPenalty: configuration.repetitionPenalty,
                    contextSize: configuration.memoryOptimization.maxKVCacheSize,
                    memoryOptimization: configuration.memoryOptimization
                )
                session?.endPhase("Generation", category: .generation)
            } else {
                // Pure MLX mode
                session?.beginPhase("Generation", category: .generation)
                tokenIds = try model.generateStream(
                    inputIds: inputs.inputIds,
                    inputFeatures: inputs.inputFeatures,
                    maxNewTokens: budget,
                    temperature: configuration.temperature,
                    topP: configuration.topP,
                    repetitionPenalty: configuration.repetitionPenalty,
                    contextSize: configuration.memoryOptimization.maxKVCacheSize,
                    memoryOptimization: configuration.memoryOptimization
                )
                session?.endPhase("Generation", category: .generation)
            }

            recordTruncation(tokenIds, budget: budget)

            // Decode tokens to text
            session?.beginPhase("Token Decoding", category: .decoding)
            let transcription = try processor.decode(tokenIds, skipSpecialTokens: true)
            session?.endPhase("Token Decoding", category: .decoding)

            return transcription
        }
        }
    }

    /// Chat with audio context
    /// - Parameters:
    ///   - audio: URL to audio file
    ///   - prompt: User prompt about the audio
    ///   - language: Optional language code. Same semantics as `transcribe(audio:language:)`
    ///     — `nil` lets the model auto-detect the spoken language.
    /// - Returns: Model response
    public func chat(audio: URL, prompt: String, language: String? = nil) async throws -> String {
        // MLX errors become VoxtralError.mlx instead of terminating the host (K-1)
        // Off the cooperative pool, cancellable at every step (K-15)
        return try await runOffCooperativePool { [self] in try withMLXErrors { _ in
            let generation = try gate.begin(
                "chat", accepts: { $0.isReady }, refusal: VoxtralPipelineError.invalidState("Model not loaded"),
                busy: VoxtralPipelineError.busy, state: .processing)
            defer {
                cachePolicy.endOfResponse()
                gate.end(generation, state: .ready)
            }

            guard let model = voxtralModel, let processor = processor else {
                throw VoxtralPipelineError.modelNotLoaded
            }

            let session = MLXProfiler.shared.activeSession
            let beacon = VoxtralRuntimeBeacon.begin(task: "chat", model: self.model.rawValue)
            defer {
                beacon?.end()
                VoxtralMemoryManager.shared.optimizeIfNeeded(tokenIndex: 0, config: configuration.memoryOptimization)
            }

            // Create chat conversation with audio
            session?.beginPhase("Audio Feature Extraction", category: .audioFeatureExtract)
            let conversation: [[String: Any]] = [
                [
                    "role": "user",
                    "content": [
                        ["type": "audio", "audio": audio.path],
                        ["type": "text", "text": prompt]
                    ]
                ]
            ]

            // Process through chat template
            guard let chatResult = try processor.applyChatTemplate(
                conversation: conversation,
                tokenize: true,
                returnTensors: "mlx"
            ) as? [String: MLXArray],
                  let inputIds = chatResult["input_ids"],
                  let inputFeatures = chatResult["input_features"] else {
                throw VoxtralPipelineError.processingFailed("Failed to process chat template")
            }
            session?.endPhase("Audio Feature Extraction", category: .audioFeatureExtract)

            // Generate response
            let budget = tokenBudget(for: audio)
            let tokenIds: [Int]

            if let hybrid = hybridEncoder, hybrid.status.coreMLAvailable {
                session?.beginPhase("Audio Encoding (CoreML)", category: .audioEncode)
                let audioEmbeds = try hybrid.encode(inputFeatures)
                session?.endPhase("Audio Encoding (CoreML)", category: .audioEncode)

                session?.beginPhase("Generation", category: .generation)
                tokenIds = try model.generateStreamWithAudioEmbeds(
                    inputIds: inputIds,
                    audioEmbeds: audioEmbeds,
                    maxNewTokens: budget,
                    temperature: configuration.temperature,
                    topP: configuration.topP,
                    repetitionPenalty: configuration.repetitionPenalty,
                    contextSize: configuration.memoryOptimization.maxKVCacheSize,
                    memoryOptimization: configuration.memoryOptimization
                )
                session?.endPhase("Generation", category: .generation)
            } else {
                session?.beginPhase("Generation", category: .generation)
                tokenIds = try model.generateStream(
                    inputIds: inputIds,
                    inputFeatures: inputFeatures,
                    maxNewTokens: budget,
                    temperature: configuration.temperature,
                    topP: configuration.topP,
                    repetitionPenalty: configuration.repetitionPenalty,
                    contextSize: configuration.memoryOptimization.maxKVCacheSize,
                    memoryOptimization: configuration.memoryOptimization
                )
                session?.endPhase("Generation", category: .generation)
            }

            recordTruncation(tokenIds, budget: budget)

            session?.beginPhase("Token Decoding", category: .decoding)
            let response = try processor.decode(tokenIds, skipSpecialTokens: true)
            session?.endPhase("Token Decoding", category: .decoding)

            return response
        }
        }
    }

    // MARK: - Cleanup

    /// Unload the model and free memory
    public func unload() {
        voxtralModel = nil
        processor = nil
        hybridEncoder = nil
        gate.reset(.unloaded)
        cachePolicy.restore()

        // Full memory cleanup
        VoxtralMemoryManager.shared.fullCleanup()
    }

    // MARK: - Utility

    /// Get current memory usage
    public var memorySummary: String {
        VoxtralMemoryManager.shared.formattedMemorySummary()
    }

    /// Check if model is ready for inference
    public var isReady: Bool {
        state.isReady
    }

    /// Get encoder status
    public var encoderStatus: String {
        if let hybrid = hybridEncoder {
            return hybrid.status.description
        }
        return "MLX encoder (GPU)"
    }
}

// MARK: - Errors

/// Pipeline-specific errors
public enum VoxtralPipelineError: Error, LocalizedError {
    case invalidState(String)
    case modelNotLoaded
    case processingFailed(String)
    case transcriptionFailed(String)
    /// Another operation (transcription, chat, loading) holds the pipeline (K-11)
    case busy(String)

    public var errorDescription: String? {
        switch self {
        case .invalidState(let message):
            return "Invalid pipeline state: \(message)"
        case .modelNotLoaded:
            return "Model not loaded. Call loadModel() first."
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .transcriptionFailed(let message):
            return "Transcription failed: \(message)"
        case .busy(let message):
            return "Pipeline busy: \(message)"
        }
    }
}

// MARK: - Convenience Extensions

extension VoxtralPipeline {

    /// Quick transcription without explicit load (loads if needed)
    /// - Parameters:
    ///   - audio: Audio file URL
    ///   - progress: Optional progress callback
    /// - Returns: Transcribed text
    public func quickTranscribe(audio: URL, progress: ProgressCallback? = nil) async throws -> String {
        if !isReady {
            try await loadModel(progress: progress)
        }
        return try await transcribe(audio: audio)
    }

    /// List available models
    static var availableModels: [Model] {
        Model.allCases
    }

    /// Get recommended model for system
    public static func recommendedModel(forRAMGB ramGB: Int? = nil) -> Model {
        let ram = ramGB ?? Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))

        switch ram {
        case 0..<16:
            return .mini3b4bit
        case 16..<32:
            return .mini3b8bit
        case 32..<64:
            return .mini3b8bit
        default:
            return .small24b8bit
        }
    }
}
