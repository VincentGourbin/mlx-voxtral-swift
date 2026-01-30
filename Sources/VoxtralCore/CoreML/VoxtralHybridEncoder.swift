/**
 * VoxtralHybridEncoder - Unified encoder supporting both Core ML (ANE) and MLX (GPU)
 *
 * This class provides a transparent interface that:
 * - Uses Core ML/ANE when available for optimal performance
 * - Falls back to MLX/GPU when Core ML is not available
 * - Handles all tensor conversion automatically
 *
 * Architecture:
 * ```
 * Input (MLXArray) --> [Core ML Encoder (ANE)] --> Output (MLXArray)
 *        |                     OR                       |
 *        +---------> [MLX Encoder (GPU)] --------------+
 * ```
 */

import Foundation
import MLX
import MLXNN

#if canImport(CoreML)
import CoreML
#endif

/// Backend selection for hybrid encoder
public enum VoxtralEncoderBackend: String, CaseIterable {
    case auto = "auto"           // Auto-select best backend
    case coreML = "coreml"       // Force Core ML (ANE)
    case mlx = "mlx"             // Force MLX (GPU)

    public var displayName: String {
        switch self {
        case .auto: return "Auto (ANE preferred)"
        case .coreML: return "Core ML (Neural Engine)"
        case .mlx: return "MLX (GPU)"
        }
    }
}

/// Status of the hybrid encoder
public struct VoxtralEncoderStatus {
    public let backend: VoxtralEncoderBackend
    public let isReady: Bool
    public let coreMLAvailable: Bool
    public let mlxAvailable: Bool
    public let lastInferenceTimeMs: Double?

    public var description: String {
        var desc = "Encoder Status:\n"
        desc += "  Backend: \(backend.displayName)\n"
        desc += "  Ready: \(isReady)\n"
        desc += "  Core ML available: \(coreMLAvailable)\n"
        desc += "  MLX available: \(mlxAvailable)\n"
        if let time = lastInferenceTimeMs {
            desc += "  Last inference: \(String(format: "%.2f", time))ms\n"
        }
        return desc
    }
}

/// Hybrid encoder supporting both Core ML and MLX backends
@available(macOS 13.0, iOS 16.0, *)
public class VoxtralHybridEncoder {

    // MARK: - Properties

    /// Core ML encoder (nil if not available)
    private var coreMLEncoder: VoxtralCoreMLEncoder?

    /// MLX encoder and projector (fallback)
    private var mlxEncoder: VoxtralEncoder?
    private var mlxProjector: VoxtralMultiModalProjector?

    /// Currently active backend
    public private(set) var activeBackend: VoxtralEncoderBackend

    /// Requested backend preference
    public var preferredBackend: VoxtralEncoderBackend {
        didSet {
            updateActiveBackend()
        }
    }

    /// Configuration for the encoder
    private let encoderConfig: VoxtralEncoderConfig
    private let projectorIntermediateSize: Int

    /// Last inference timing for monitoring
    private var lastInferenceTimeMs: Double?

    // MARK: - Initialization

    /// Initialize hybrid encoder with optional Core ML model URL
    /// - Parameters:
    ///   - coreMLModelURL: Optional URL to Core ML model
    ///   - coreMLConfig: Optional Core ML configuration with variant info
    ///   - encoderConfig: Configuration for MLX fallback encoder
    ///   - projectorIntermediateSize: Intermediate size for projector
    ///   - preferredBackend: Preferred backend (default: auto)
    public init(
        coreMLModelURL: URL? = nil,
        coreMLConfig: VoxtralCoreMLConfig? = nil,
        encoderConfig: VoxtralEncoderConfig = VoxtralEncoderConfig(),
        projectorIntermediateSize: Int = 5120,
        preferredBackend: VoxtralEncoderBackend = .auto
    ) {
        self.encoderConfig = encoderConfig
        self.projectorIntermediateSize = projectorIntermediateSize
        self.preferredBackend = preferredBackend
        self.activeBackend = .mlx  // Will be updated

        // Try to load Core ML encoder
        if let url = coreMLModelURL {
            do {
                self.coreMLEncoder = try VoxtralCoreMLEncoder(modelURL: url)
                VoxtralDebug.log("Core ML encoder loaded successfully")
            } catch {
                VoxtralDebug.log("Failed to load Core ML encoder: \(error)")
            }
        } else {
            // Try auto-discovery with variant-specific config
            let config = coreMLConfig ?? .default
            do {
                self.coreMLEncoder = try VoxtralCoreMLEncoder(config: config)
                VoxtralDebug.log("Core ML encoder auto-discovered (variant: \(config.variant.rawValue))")
            } catch {
                VoxtralDebug.log("Core ML encoder not found, using MLX fallback")
            }
        }

        // Initialize MLX fallback
        self.mlxEncoder = VoxtralEncoder(config: encoderConfig)
        // Note: Projector will be set separately via setMLXProjector()

        // Select active backend
        updateActiveBackend()
    }

    /// Set the MLX projector for fallback mode
    /// - Parameter projector: The VoxtralMultiModalProjector instance
    public func setMLXProjector(_ projector: VoxtralMultiModalProjector) {
        self.mlxProjector = projector
    }

    /// Set the MLX encoder for fallback mode (with loaded weights)
    /// - Parameter encoder: The VoxtralEncoder instance with loaded weights
    public func setMLXEncoder(_ encoder: VoxtralEncoder) {
        self.mlxEncoder = encoder
    }

    /// Set the MLX encoder from VoxtralStandardEncoder (with loaded weights)
    /// - Parameter encoder: The VoxtralStandardEncoder instance with loaded weights
    public func setMLXEncoderFromStandard(_ encoder: VoxtralStandardEncoder) {
        // Store reference to use in encodeMLX
        self.standardEncoder = encoder
    }

    /// Set the MLX projector from VoxtralStandardProjector (with loaded weights)
    /// - Parameter projector: The VoxtralStandardProjector instance with loaded weights
    public func setMLXProjectorFromStandard(_ projector: VoxtralStandardProjector) {
        // Store reference to use in encodeMLX
        self.standardProjector = projector
    }

    /// Internal reference to standard encoder for fallback
    private var standardEncoder: VoxtralStandardEncoder?

    /// Internal reference to standard projector for fallback
    private var standardProjector: VoxtralStandardProjector?

    // MARK: - Backend Selection

    private func updateActiveBackend() {
        switch preferredBackend {
        case .auto:
            // Prefer Core ML if available
            if coreMLEncoder != nil {
                activeBackend = .coreML
            } else {
                activeBackend = .mlx
            }
        case .coreML:
            if coreMLEncoder != nil {
                activeBackend = .coreML
            } else {
                VoxtralDebug.log("Core ML requested but not available, falling back to MLX")
                activeBackend = .mlx
            }
        case .mlx:
            activeBackend = .mlx
        }

        VoxtralDebug.log("Encoder backend: \(activeBackend.displayName)")
    }

    // MARK: - Encoding

    /// Encode audio features to embeddings
    /// - Parameter inputFeatures: Mel spectrogram [numChunks, 128, 3000]
    /// - Returns: Audio embeddings [1, numFrames, hiddenSize]
    public func encode(_ inputFeatures: MLXArray) throws -> MLXArray {
        let startTime = CFAbsoluteTimeGetCurrent()

        let result: MLXArray

        switch activeBackend {
        case .coreML, .auto:
            if let coreML = coreMLEncoder {
                result = try encodeCoreML(inputFeatures, encoder: coreML)
            } else {
                result = try encodeMLX(inputFeatures)
            }
        case .mlx:
            result = try encodeMLX(inputFeatures)
        }

        lastInferenceTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

        return result
    }

    /// Encode using Core ML (ANE)
    private func encodeCoreML(_ inputFeatures: MLXArray, encoder: VoxtralCoreMLEncoder) throws -> MLXArray {
        // Input: [numChunks, 128, 3000]
        let numChunks = inputFeatures.shape[0]

        var allEmbeddings: [MLXArray] = []

        // Process each chunk through Core ML
        // Note: Core ML expects [1, 128, 3000] per inference
        for chunkIdx in 0..<numChunks {
            // Extract single chunk
            let chunk = inputFeatures[chunkIdx]  // [128, 3000]
            let batchedChunk = expandedDimensions(chunk, axis: 0)  // [1, 128, 3000]

            // Convert to MLMultiArray
            let mlMultiArray = try batchedChunk.toMLMultiArray()

            // Run Core ML inference
            let embeddings = try encoder.encode(mlMultiArray)

            // Convert back to MLXArray
            let mlxEmbeddings = embeddings.toMLXArray()  // [1, 375, 3072]

            allEmbeddings.append(mlxEmbeddings.squeezed(axis: 0))  // [375, 3072]
        }

        // Concatenate all chunk embeddings
        let concatenated = concatenated(allEmbeddings, axis: 0)  // [numChunks * 375, 3072]

        // Add batch dimension
        let result = expandedDimensions(concatenated, axis: 0)  // [1, numChunks * 375, 3072]

        eval(result)
        return result
    }

    /// Encode using MLX (GPU) - fallback
    private func encodeMLX(_ inputFeatures: MLXArray) throws -> MLXArray {
        // Prefer standard encoder with loaded weights if available
        if let stdEncoder = standardEncoder {
            // Use the loaded standard encoder and projector
            VoxtralDebug.log("Using standard encoder with loaded weights")
            // VoxtralStandardEncoder returns MLXArray directly (not a tuple)
            let hiddenStates = stdEncoder(inputFeatures)  // [numChunks, 1500, 1280]

            // Reshape for projector: [numChunks, 1500, 1280] -> [-1, 5120]
            let reshaped = hiddenStates.reshaped([-1, projectorIntermediateSize])

            // Project using standard projector with loaded weights (preferred)
            let projected: MLXArray
            if let stdProjector = standardProjector {
                projected = stdProjector(reshaped)  // [-1, 3072]
                VoxtralDebug.log("Using standard projector with loaded weights")
            } else if let projector = mlxProjector {
                projected = projector(reshaped)  // [-1, 3072]
                VoxtralDebug.log("WARNING: Using empty projector (no loaded weights)")
            } else {
                VoxtralDebug.log("WARNING: No projector set, returning raw encoder output")
                projected = reshaped
            }

            // Add batch dimension
            let result = expandedDimensions(projected, axis: 0)
            eval(result)
            return result
        }

        // Fallback to basic MLX encoder (uninitialized weights - not recommended)
        guard let encoder = mlxEncoder else {
            throw VoxtralCoreMLError.notAvailable("MLX encoder not initialized")
        }

        VoxtralDebug.log("WARNING: Using MLX encoder without loaded weights")

        // Process through encoder
        let (hiddenStates, _, _) = encoder(inputFeatures)  // [numChunks, 1500, 1280]

        // Reshape for projector: [numChunks, 1500, 1280] -> [-1, 5120]
        let reshaped = hiddenStates.reshaped([-1, projectorIntermediateSize])

        // Project if projector is available
        let projected: MLXArray
        if let projector = mlxProjector {
            projected = projector(reshaped)  // [-1, 3072]
        } else {
            // Without projector, just return reshaped (may not work correctly)
            VoxtralDebug.log("WARNING: No projector set, returning raw encoder output")
            projected = reshaped
        }

        // Add batch dimension
        let result = expandedDimensions(projected, axis: 0)

        eval(result)
        return result
    }

    // MARK: - Status

    /// Get current encoder status
    public var status: VoxtralEncoderStatus {
        VoxtralEncoderStatus(
            backend: activeBackend,
            isReady: coreMLEncoder != nil || mlxEncoder != nil,
            coreMLAvailable: coreMLEncoder != nil,
            mlxAvailable: mlxEncoder != nil,
            lastInferenceTimeMs: lastInferenceTimeMs
        )
    }

    /// Print status to debug log
    public func logStatus() {
        VoxtralDebug.log(status.description)
    }

    // MARK: - Benchmarking

    /// Benchmark encoder performance
    /// - Parameters:
    ///   - iterations: Number of iterations
    ///   - warmup: Warmup iterations
    /// - Returns: Dictionary of backend -> average time in ms
    public func benchmark(iterations: Int = 10, warmup: Int = 3) throws -> [VoxtralEncoderBackend: Double] {
        var results: [VoxtralEncoderBackend: Double] = [:]

        // Create test input
        let testInput = MLXArray.zeros([1, 128, 3000])

        // Benchmark Core ML if available
        if coreMLEncoder != nil {
            let originalBackend = activeBackend
            activeBackend = .coreML

            // Warmup
            for _ in 0..<warmup {
                _ = try encode(testInput)
            }

            // Benchmark
            var totalTime: Double = 0
            for _ in 0..<iterations {
                _ = try encode(testInput)
                totalTime += lastInferenceTimeMs ?? 0
            }

            results[.coreML] = totalTime / Double(iterations)
            activeBackend = originalBackend
        }

        // Benchmark MLX
        if mlxEncoder != nil {
            let originalBackend = activeBackend
            activeBackend = .mlx

            // Warmup
            for _ in 0..<warmup {
                _ = try encode(testInput)
            }

            // Benchmark
            var totalTime: Double = 0
            for _ in 0..<iterations {
                _ = try encode(testInput)
                totalTime += lastInferenceTimeMs ?? 0
            }

            results[.mlx] = totalTime / Double(iterations)
            activeBackend = originalBackend
        }

        // Log results
        VoxtralDebug.log("\nEncoder Benchmark Results:")
        for (backend, time) in results.sorted(by: { $0.value < $1.value }) {
            VoxtralDebug.log("  \(backend.displayName): \(String(format: "%.2f", time))ms")
        }

        return results
    }
}

// MARK: - Factory Methods

@available(macOS 13.0, iOS 16.0, *)
extension VoxtralHybridEncoder {

    /// Create hybrid encoder with Core ML downloaded from HuggingFace
    /// - Parameters:
    ///   - variant: Core ML variant to download (mini or small)
    ///   - encoderConfig: Configuration for MLX fallback encoder
    ///   - projectorIntermediateSize: Intermediate size for projector
    ///   - preferredBackend: Preferred backend (default: auto)
    ///   - progress: Optional download progress callback
    /// - Returns: Configured VoxtralHybridEncoder
    public static func withHuggingFaceDownload(
        variant: VoxtralCoreMLVariant = .mini,
        encoderConfig: VoxtralEncoderConfig = VoxtralEncoderConfig(),
        projectorIntermediateSize: Int = 5120,
        preferredBackend: VoxtralEncoderBackend = .auto,
        progress: ((Double, String) -> Void)? = nil
    ) async throws -> VoxtralHybridEncoder {
        // Download Core ML model from HuggingFace
        let modelURL = try await VoxtralCoreMLEncoder.downloadFromHuggingFace(
            variant: variant,
            progress: progress
        )

        // Create hybrid encoder with downloaded model
        return VoxtralHybridEncoder(
            coreMLModelURL: modelURL,
            encoderConfig: encoderConfig,
            projectorIntermediateSize: projectorIntermediateSize,
            preferredBackend: preferredBackend
        )
    }

    /// Create hybrid encoder with Core ML auto-selected for MLX model
    /// - Parameters:
    ///   - mlxModelRepoId: MLX model repository ID to match variant
    ///   - encoderConfig: Configuration for MLX fallback encoder
    ///   - projectorIntermediateSize: Intermediate size for projector
    ///   - preferredBackend: Preferred backend (default: auto)
    ///   - progress: Optional download progress callback
    /// - Returns: Configured VoxtralHybridEncoder with matching Core ML variant
    public static func forMLXModel(
        mlxModelRepoId: String,
        encoderConfig: VoxtralEncoderConfig = VoxtralEncoderConfig(),
        projectorIntermediateSize: Int = 5120,
        preferredBackend: VoxtralEncoderBackend = .auto,
        progress: ((Double, String) -> Void)? = nil
    ) async throws -> VoxtralHybridEncoder {
        let variant = VoxtralCoreMLVariant.fromMLXModelRepoId(mlxModelRepoId)
        VoxtralDebug.log("Auto-selected Core ML variant '\(variant.rawValue)' for MLX model: \(mlxModelRepoId)")

        return try await withHuggingFaceDownload(
            variant: variant,
            encoderConfig: encoderConfig,
            projectorIntermediateSize: projectorIntermediateSize,
            preferredBackend: preferredBackend,
            progress: progress
        )
    }
}

// MARK: - Integration with VoxtralForConditionalGeneration

@available(macOS 13.0, iOS 16.0, *)
extension VoxtralForConditionalGeneration {

    /// Create a hybrid encoder for this model
    /// - Parameter preferredBackend: Preferred backend for audio encoding
    /// - Returns: Configured VoxtralHybridEncoder
    public func createHybridEncoder(preferredBackend: VoxtralEncoderBackend = .auto) -> VoxtralHybridEncoder {
        // Determine Core ML variant based on text hidden size
        // Small (24B) has hiddenSize 5120, Mini (3B) has hiddenSize 3072
        let coreMLVariant: VoxtralCoreMLVariant = config.textConfig.hiddenSize == 5120 ? .small : .mini
        VoxtralDebug.log("Model text hiddenSize: \(config.textConfig.hiddenSize) -> Core ML variant: \(coreMLVariant.rawValue)")

        let coreMLConfig = coreMLVariant == .small ? VoxtralCoreMLConfig.small : VoxtralCoreMLConfig.mini

        let encoderConfig = VoxtralEncoderConfig(
            hidden_size: config.audioConfig.hiddenSize,
            intermediate_size: config.audioConfig.intermediate_size,
            num_hidden_layers: 32,  // Standard for Voxtral
            num_attention_heads: config.audioConfig.numAttentionHeads
        )

        let hybridEncoder = VoxtralHybridEncoder(
            coreMLConfig: coreMLConfig,
            encoderConfig: encoderConfig,
            projectorIntermediateSize: config.audioConfig.intermediate_size,
            preferredBackend: preferredBackend
        )

        // Set the MLX projector for fallback (empty - not recommended)
        hybridEncoder.setMLXProjector(multiModalProjector)

        // Set the standard encoder and projector with loaded weights for MLX fallback
        if let stdModel = standardModel {
            hybridEncoder.setMLXEncoderFromStandard(stdModel.audioTower)
            hybridEncoder.setMLXProjectorFromStandard(stdModel.multiModalProjector)
            VoxtralDebug.log("Hybrid encoder: MLX fallback will use loaded audio tower and projector weights")
        }

        return hybridEncoder
    }

    /// Create a hybrid encoder with Core ML downloaded from HuggingFace
    /// The variant is auto-selected based on this model's configuration
    /// - Parameters:
    ///   - preferredBackend: Preferred backend for audio encoding
    ///   - progress: Optional download progress callback
    /// - Returns: Configured VoxtralHybridEncoder with matching Core ML model
    public func createHybridEncoderWithDownload(
        preferredBackend: VoxtralEncoderBackend = .auto,
        progress: ((Double, String) -> Void)? = nil
    ) async throws -> VoxtralHybridEncoder {
        // Determine variant based on text hidden size
        let variant: VoxtralCoreMLVariant = config.textConfig.hiddenSize == 5120 ? .small : .mini
        VoxtralDebug.log("Model text hiddenSize: \(config.textConfig.hiddenSize) -> Core ML variant: \(variant.rawValue)")

        let encoderConfig = VoxtralEncoderConfig(
            hidden_size: config.audioConfig.hiddenSize,
            intermediate_size: config.audioConfig.intermediate_size,
            num_hidden_layers: 32,
            num_attention_heads: config.audioConfig.numAttentionHeads
        )

        let hybridEncoder = try await VoxtralHybridEncoder.withHuggingFaceDownload(
            variant: variant,
            encoderConfig: encoderConfig,
            projectorIntermediateSize: config.audioConfig.intermediate_size,
            preferredBackend: preferredBackend,
            progress: progress
        )

        // Set the MLX projector for fallback
        hybridEncoder.setMLXProjector(multiModalProjector)

        // Set the standard encoder and projector with loaded weights for MLX fallback
        if let stdModel = standardModel {
            hybridEncoder.setMLXEncoderFromStandard(stdModel.audioTower)
            hybridEncoder.setMLXProjectorFromStandard(stdModel.multiModalProjector)
            VoxtralDebug.log("Hybrid encoder: MLX fallback will use loaded audio tower and projector weights")
        }

        return hybridEncoder
    }
}
