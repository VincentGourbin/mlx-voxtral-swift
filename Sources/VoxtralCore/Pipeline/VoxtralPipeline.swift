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

import Foundation
import MLX
import MLXNN

/// Simplified facade for Voxtral speech-to-text
@available(macOS 13.0, iOS 16.0, *)
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

        /// Get the HuggingFace repo ID for this model
        public var repoId: String {
            switch self {
            case .mini3b:
                return "mistralai/Voxtral-Mini-3B-2507"
            case .mini3b8bit:
                return "mzbac/voxtral-mini-3b-8bit"
            case .mini3b4bit:
                return "mzbac/voxtral-mini-3b-4bit-mixed"
            case .small24b:
                return "mistralai/Voxtral-Small-24B-2507"
            case .small24b8bit:
                return "mzbac/Voxtral-Small-24B-2507-8bit"
            case .small4bit:
                return "VincentGOURBIN/voxtral-small-4bit-mixed"
            }
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
        /// Maximum tokens to generate
        public var maxTokens: Int

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
                maxTokens: 500,
                temperature: 0.0,
                topP: 0.95,
                repetitionPenalty: 1.2,
                memoryOptimization: .recommended()
            )
        }

        public init(
            maxTokens: Int = 500,
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
    public private(set) var state: State = .unloaded

    /// Loaded Voxtral model
    private var voxtralModel: VoxtralModel?

    /// Loaded processor
    private var processor: VoxtralProcessor?

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

        // Apply memory optimization settings
        VoxtralMemoryManager.shared.config = configuration.memoryOptimization
    }

    // MARK: - Model Loading

    /// Load the model
    /// - Parameter progress: Optional progress callback (progress 0-1, status message)
    public func loadModel(progress: ProgressCallback? = nil) async throws {
        guard state.isUnloaded || state.isError else {
            throw VoxtralPipelineError.invalidState("Model already loaded or loading")
        }

        state = .loading
        progress?(0.0, "Starting model download...")

        do {
            // Download/resolve model path
            progress?(0.1, "Downloading model...")
            let modelPath = try await ModelDownloader.resolveModel(model.repoId) { downloadProgress, status in
                progress?(0.1 + downloadProgress * 0.4, status)
            }

            // Load model
            progress?(0.5, "Loading model...")
            let (loadedModel, _) = try loadVoxtralModel(
                modelPath: modelPath.path,
                dtype: .bfloat16,
                lazy: false
            )
            self.voxtralModel = loadedModel

            // Load processor (includes tokenizer loading which can be slow)
            progress?(0.6, "Loading tokenizer...")
            self.processor = try VoxtralProcessor.fromPretrained(modelPath.path) { processorProgress, status in
                // Map processor progress (0-1) to pipeline progress (0.6-0.85)
                progress?(0.6 + processorProgress * 0.25, status)
            }

            // Setup hybrid encoder if needed
            progress?(0.85, "Configuring encoder...")
            try setupEncoder()

            state = .ready
            progress?(1.0, "Model ready!")

        } catch {
            state = .error(error.localizedDescription)
            throw error
        }
    }

    /// Setup encoder based on backend preference
    private func setupEncoder() throws {
        guard let voxtralModel = voxtralModel as? VoxtralForConditionalGeneration else {
            return
        }

        switch backend {
        case .hybrid, .auto:
            // Try to create hybrid encoder with Core ML
            hybridEncoder = voxtralModel.createHybridEncoder(
                preferredBackend: backend == .hybrid ? .coreML : .auto
            )
            VoxtralDebug.log("Hybrid encoder created: \(hybridEncoder?.status.description ?? "nil")")

        case .mlx:
            // Pure MLX mode - no hybrid encoder needed
            hybridEncoder = nil
        }
    }

    // MARK: - Transcription

    /// Transcribe audio file
    /// - Parameters:
    ///   - audio: URL to audio file
    ///   - language: Language code (default: "en")
    /// - Returns: Transcribed text
    public func transcribe(audio: URL, language: String = "en") async throws -> String {
        guard state.isReady else {
            throw VoxtralPipelineError.invalidState("Model not loaded")
        }

        guard let model = voxtralModel, let processor = processor else {
            throw VoxtralPipelineError.modelNotLoaded
        }

        state = .processing
        defer {
            state = .ready
            // Apply memory optimization
            VoxtralMemoryManager.shared.optimizeIfNeeded(tokenIndex: 0)
        }

        // Create transcription request (note: method name has typo in original)
        let inputs = try processor.applyTranscritionRequest(
            audio: audio.path,
            language: language
        )

        // Generate transcription
        let tokenIds: [Int]

        if let hybrid = hybridEncoder, hybrid.status.coreMLAvailable {
            // Hybrid mode: use Core ML for audio encoding
            let audioEmbeds = try hybrid.encode(inputs.inputFeatures)
            tokenIds = try model.generateStreamWithAudioEmbeds(
                inputIds: inputs.inputIds,
                audioEmbeds: audioEmbeds,
                maxNewTokens: configuration.maxTokens,
                temperature: configuration.temperature,
                topP: configuration.topP,
                repetitionPenalty: configuration.repetitionPenalty,
                contextSize: configuration.memoryOptimization.maxKVCacheSize
            )
        } else {
            // Pure MLX mode
            tokenIds = try model.generateStream(
                inputIds: inputs.inputIds,
                inputFeatures: inputs.inputFeatures,
                maxNewTokens: configuration.maxTokens,
                temperature: configuration.temperature,
                topP: configuration.topP,
                repetitionPenalty: configuration.repetitionPenalty,
                contextSize: configuration.memoryOptimization.maxKVCacheSize
            )
        }

        // Decode tokens to text
        let transcription = try processor.decode(tokenIds, skipSpecialTokens: true)

        return transcription
    }

    /// Chat with audio context
    /// - Parameters:
    ///   - audio: URL to audio file
    ///   - prompt: User prompt about the audio
    ///   - language: Language code (default: "en")
    /// - Returns: Model response
    public func chat(audio: URL, prompt: String, language: String = "en") async throws -> String {
        guard state.isReady else {
            throw VoxtralPipelineError.invalidState("Model not loaded")
        }

        guard let model = voxtralModel, let processor = processor else {
            throw VoxtralPipelineError.modelNotLoaded
        }

        state = .processing
        defer {
            state = .ready
            VoxtralMemoryManager.shared.optimizeIfNeeded(tokenIndex: 0)
        }

        // Create chat conversation with audio
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

        // Generate response
        let tokenIds: [Int]

        if let hybrid = hybridEncoder, hybrid.status.coreMLAvailable {
            let audioEmbeds = try hybrid.encode(inputFeatures)
            tokenIds = try model.generateStreamWithAudioEmbeds(
                inputIds: inputIds,
                audioEmbeds: audioEmbeds,
                maxNewTokens: configuration.maxTokens,
                temperature: configuration.temperature,
                topP: configuration.topP,
                repetitionPenalty: configuration.repetitionPenalty,
                contextSize: configuration.memoryOptimization.maxKVCacheSize
            )
        } else {
            tokenIds = try model.generateStream(
                inputIds: inputIds,
                inputFeatures: inputFeatures,
                maxNewTokens: configuration.maxTokens,
                temperature: configuration.temperature,
                topP: configuration.topP,
                repetitionPenalty: configuration.repetitionPenalty,
                contextSize: configuration.memoryOptimization.maxKVCacheSize
            )
        }

        let response = try processor.decode(tokenIds, skipSpecialTokens: true)

        return response
    }

    // MARK: - Cleanup

    /// Unload the model and free memory
    public func unload() {
        voxtralModel = nil
        processor = nil
        hybridEncoder = nil
        state = .unloaded

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
        }
    }
}

// MARK: - Convenience Extensions

@available(macOS 13.0, iOS 16.0, *)
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
    public static var availableModels: [Model] {
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
