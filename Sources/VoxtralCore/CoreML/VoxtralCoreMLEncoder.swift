/**
 * VoxtralCoreMLEncoder - Core ML wrapper for Voxtral audio encoder
 *
 * This class provides a Swift interface to run the Voxtral audio encoder
 * on Apple Neural Engine (ANE) via Core ML for optimized performance.
 *
 * Benefits over MLX GPU:
 * - 2-3x faster inference on Neural Engine
 * - Lower power consumption
 * - Reduced thermal throttling
 *
 * Requirements:
 * - macOS 13.0+ / iOS 16.0+
 * - VoxtralEncoder.mlpackage in app bundle
 */

import Foundation
@preconcurrency import CoreML

/// Errors specific to Core ML encoder operations
public enum VoxtralCoreMLError: Error, LocalizedError {
    case modelNotFound(String)
    case modelLoadFailed(String)
    case inputCreationFailed(String)
    case predictionFailed(String)
    case outputExtractionFailed(String)
    case invalidInputShape([Int], expected: [Int])
    case notAvailable(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Core ML model not found at: \(path)"
        case .modelLoadFailed(let reason):
            return "Failed to load Core ML model: \(reason)"
        case .inputCreationFailed(let reason):
            return "Failed to create model input: \(reason)"
        case .predictionFailed(let reason):
            return "Model prediction failed: \(reason)"
        case .outputExtractionFailed(let reason):
            return "Failed to extract model output: \(reason)"
        case .invalidInputShape(let shape, let expected):
            return "Invalid input shape \(shape), expected \(expected)"
        case .notAvailable(let reason):
            return "Core ML encoder not available: \(reason)"
        }
    }
}

/// Voxtral Core ML model variant
public enum VoxtralCoreMLVariant: String, Sendable {
    case mini = "mini"    // 3B - output 3072
    case small = "small"  // 24B - output 5120

    /// Output hidden size for this variant
    public var hiddenSize: Int {
        switch self {
        case .mini: return 3072
        case .small: return 5120
        }
    }

    /// HuggingFace repository for this variant
    public var huggingFaceRepo: String {
        switch self {
        case .mini: return "VincentGOURBIN/voxtral-encoder-coreml-mini"
        case .small: return "VincentGOURBIN/voxtral-encoder-coreml-small"
        }
    }

    /// Model file name for this variant
    public var modelName: String {
        switch self {
        case .mini: return "VoxtralEncoderMini.mlmodelc"
        case .small: return "VoxtralEncoderSmall.mlmodelc"
        }
    }

    /// Variant from a model folder's `config.json`: `text_config.hidden_size` 5120 → Small, 3072 → Mini
    /// (K-24: the name alone misreads a Small folder called e.g. `x/model`); nil without a readable config
    public static func variant(forConfigAt folder: URL) -> VoxtralCoreMLVariant? {
        guard let data = try? Data(contentsOf: folder.appendingPathComponent("config.json")),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let text = json["text_config"] as? [String: Any],
              let hidden = text["hidden_size"] as? Int else { return nil }
        switch hidden {
        case 5120: return .small
        case 3072: return .mini
        default: return nil
        }
    }

    /// Detect variant from MLX model repo ID, or from its `config.json` when the id is a local folder (K-24)
    public static func fromMLXModelRepoId(_ repoId: String) -> VoxtralCoreMLVariant {
        if let fromConfig = variant(forConfigAt: URL(fileURLWithPath: repoId)) {
            return fromConfig
        }
        // Small models (24B)
        if repoId.lowercased().contains("small") || repoId.lowercased().contains("24b") {
            return .small
        }
        // Default to mini (3B)
        return .mini
    }

    /// Description for display
    public var description: String {
        switch self {
        case .mini: return "Mini (3B) - output [1, 375, 3072]"
        case .small: return "Small (24B) - output [1, 375, 5120]"
        }
    }
}

/// Configuration for Core ML encoder
struct VoxtralCoreMLConfig {
    /// Model variant (mini or small)
    var variant: VoxtralCoreMLVariant

    /// Preferred compute units (default: cpuAndNeuralEngine for ANE)
    var computeUnits: MLComputeUnits

    /// Whether to allow low precision accumulation (faster but less precise)
    var allowLowPrecisionAccumulationOnGPU: Bool

    /// Expected input shape [batch, melBins, frames]
    let inputShape: [Int] = [1, 128, 3000]

    /// Expected output shape [batch, audioFrames, hiddenSize] - depends on variant
    var outputShape: [Int] {
        [1, 375, variant.hiddenSize]
    }

    /// Default configuration: Mini variant on the Neural Engine. On the GPU (MPSGraph) Core ML deadlocks
    /// with MLX in the same process: its first prediction waits forever for a Metal command buffer
    /// (K-32b, decided by Vincent on 2026-10-01; ANE output identical to MLX on C-moyen)
    static var `default`: VoxtralCoreMLConfig {
        VoxtralCoreMLConfig(
            variant: .mini,
            computeUnits: .cpuAndNeuralEngine,
            allowLowPrecisionAccumulationOnGPU: true
        )
    }

    /// Default configuration for Mini variant (Neural Engine, see `default`)
    static var mini: VoxtralCoreMLConfig {
        VoxtralCoreMLConfig(
            variant: .mini,
            computeUnits: .cpuAndNeuralEngine,
            allowLowPrecisionAccumulationOnGPU: true
        )
    }

    /// Default configuration for Small variant (Neural Engine, see `default`)
    static var small: VoxtralCoreMLConfig {
        VoxtralCoreMLConfig(
            variant: .small,
            computeUnits: .cpuAndNeuralEngine,
            allowLowPrecisionAccumulationOnGPU: true
        )
    }

    /// Configuration for GPU-only execution. Deadlocks when MLX runs in the same process (the STT
    /// pipeline does): use only in a process without MLX (K-32b)
    static var gpuOnly: VoxtralCoreMLConfig {
        VoxtralCoreMLConfig(
            variant: .mini,
            computeUnits: .cpuAndGPU,
            allowLowPrecisionAccumulationOnGPU: true
        )
    }

    /// Configuration for CPU-only execution (fallback)
    static var cpuOnly: VoxtralCoreMLConfig {
        VoxtralCoreMLConfig(
            variant: .mini,
            computeUnits: .cpuOnly,
            allowLowPrecisionAccumulationOnGPU: false
        )
    }

    init(
        variant: VoxtralCoreMLVariant = .mini,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        allowLowPrecisionAccumulationOnGPU: Bool = true
    ) {
        self.variant = variant
        self.computeUnits = computeUnits
        self.allowLowPrecisionAccumulationOnGPU = allowLowPrecisionAccumulationOnGPU
    }
}

/// Core ML wrapper for Voxtral audio encoder
public class VoxtralCoreMLEncoder: @unchecked Sendable {

    // MARK: - Static Configuration


    // MARK: - Properties

    /// The loaded Core ML model
    private let model: MLModel

    /// Configuration used for this encoder
    let config: VoxtralCoreMLConfig

    /// Whether the model is loaded and ready
    var isReady: Bool { true }

    /// Model input name
    private let inputName = "mel_spectrogram"

    /// Model output name
    private let outputName = "audio_embeddings"

    // MARK: - Initialization

    /// Initialize with a model URL and configuration
    /// - Parameters:
    ///   - modelURL: URL to the .mlpackage or .mlmodelc
    ///   - config: Configuration for compute units and options
    init(modelURL: URL, config: VoxtralCoreMLConfig = .default) throws {
        self.config = config

        // Create MLModel configuration
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits
        mlConfig.allowLowPrecisionAccumulationOnGPU = config.allowLowPrecisionAccumulationOnGPU

        // Load the model
        do {
            self.model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
            VoxtralDebug.log("Core ML encoder loaded from: \(modelURL.lastPathComponent)")
            VoxtralDebug.log("Compute units: \(config.computeUnits)")
        } catch {
            throw VoxtralCoreMLError.modelLoadFailed(error.localizedDescription)
        }

        // A mini encoder (3072) under a small config (5120), or the reverse, would produce
        // embeddings of the wrong width: refuse it here rather than at the first transcription.
        if let declared = model.modelDescription.outputDescriptionsByName[outputName]?
            .multiArrayConstraint?.shape.map({ $0.intValue }),
           let width = declared.last, width != config.variant.hiddenSize {
            throw VoxtralCoreMLError.modelLoadFailed(
                "\(modelURL.lastPathComponent) outputs \(declared), expected width \(config.variant.hiddenSize) for the \(config.variant.rawValue) variant")
        }
    }

    /// Initialize by searching for the model in common locations
    /// - Parameter config: Configuration for compute units and options
    convenience init(config: VoxtralCoreMLConfig = .default) throws {
        // Search order:
        // 1. App bundle
        // 2. Documents directory
        // 3. Current directory

        // Build search list based on variant (variant-specific models first)
        var modelNames: [String] = []

        // Add variant-specific model names first
        modelNames.append(config.variant.modelName)
        modelNames.append(config.variant.modelName.replacingOccurrences(of: ".mlmodelc", with: ".mlpackage"))

        // Generic legacy names are mini encoders (3072): never offer them to another variant
        if config.variant == .mini {
            modelNames.append(contentsOf: [
                "VoxtralEncoderFull.mlmodelc",
                "VoxtralEncoderFull.mlpackage",
                "VoxtralEncoder.mlmodelc",
                "VoxtralEncoder.mlpackage"
            ])
        }
        var foundURL: URL?

        // Search in main bundle
        for name in modelNames {
            let baseName = URL(fileURLWithPath: name).deletingPathExtension().lastPathComponent
            let ext = URL(fileURLWithPath: name).pathExtension
            if let url = Bundle.main.url(forResource: baseName, withExtension: ext) {
                foundURL = url
                VoxtralDebug.log("[CoreML] Found in main bundle: \(url.path)")
                break
            }
        }

        // Search in Documents
        if foundURL == nil {
            let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            for name in modelNames {
                let url = documentsURL.appendingPathComponent("VoxtralModels").appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: url.path) {
                    foundURL = url
                    break
                }
            }
        }

        // Search in Resources directory (relative to executable for CLI)
        if foundURL == nil {
            // Executable-relative Resources (CLI and SwiftPM builds) and VOXTRAL_RESOURCES_PATH; no lookup in the
            // current directory nor in a developer's home (K-23)
            let executableURL = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            let executableDir = executableURL.deletingLastPathComponent()
            var possiblePaths = (0 ..< 4).map { level in
                (0 ..< level).reduce(executableDir) { url, _ in url.deletingLastPathComponent() }
                    .appendingPathComponent("Resources")
            }
            if let envPath = ProcessInfo.processInfo.environment["VOXTRAL_RESOURCES_PATH"] {
                possiblePaths.insert(URL(fileURLWithPath: envPath), at: 0)
            }

            VoxtralDebug.log("[CoreML] Searching for Core ML model in \(possiblePaths.count) paths (executable: \(executableURL.path))")
            for basePath in possiblePaths {
                VoxtralDebug.log("[CoreML]   Checking: \(basePath.path)")
                for name in modelNames {
                    let url = basePath.appendingPathComponent(name)
                    if FileManager.default.fileExists(atPath: url.path) {
                        foundURL = url
                        VoxtralDebug.log("Found Core ML model at: \(url.path)")
                        break
                    }
                }
                if foundURL != nil { break }
            }
        }

        guard let modelURL = foundURL else {
            let searchedPaths = "Searched in: CWD/Resources, executable-relative paths"
            throw VoxtralCoreMLError.modelNotFound("VoxtralEncoder.mlpackage not found. \(searchedPaths). Set VOXTRAL_RESOURCES_PATH env var to specify location.")
        }

        try self.init(modelURL: modelURL, config: config)
    }

    // MARK: - Public API

    /// Encode audio mel-spectrogram to embeddings
    /// - Parameter melSpectrogram: Mel spectrogram as MLMultiArray [1, 128, 3000]
    /// - Returns: Audio embeddings as MLMultiArray [1, 375, 3072]
    func encode(_ melSpectrogram: MLMultiArray) throws -> MLMultiArray {
        // Validate input shape
        let shape = melSpectrogram.shape.map { $0.intValue }
        guard shape == config.inputShape else {
            throw VoxtralCoreMLError.invalidInputShape(shape, expected: config.inputShape)
        }

        // Create input provider
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: melSpectrogram
        ])

        // Run prediction
        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: inputProvider)
        } catch {
            throw VoxtralCoreMLError.predictionFailed(error.localizedDescription)
        }

        // Extract output
        guard let audioEmbeddings = output.featureValue(for: outputName)?.multiArrayValue else {
            throw VoxtralCoreMLError.outputExtractionFailed("Output '\(outputName)' not found or not MLMultiArray")
        }
        let outputShape = audioEmbeddings.shape.map { $0.intValue }
        guard outputShape == config.outputShape else {
            throw VoxtralCoreMLError.outputExtractionFailed("Output shape \(outputShape), expected \(config.outputShape)")
        }

        return audioEmbeddings
    }

    /// Encode audio mel-spectrogram from Float array
    /// - Parameters:
    ///   - melData: Flat Float array of mel spectrogram data
    ///   - shape: Shape of the data [batch, melBins, frames]
    /// - Returns: Audio embeddings as MLMultiArray [1, 375, 3072]
    func encode(_ melData: [Float], shape: [Int]) throws -> MLMultiArray {
        guard shape == config.inputShape else {
            throw VoxtralCoreMLError.invalidInputShape(shape, expected: config.inputShape)
        }

        // Create MLMultiArray from Float array
        let multiArray = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)

        // Copy data
        let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        for (index, value) in melData.enumerated() {
            pointer[index] = value
        }

        return try encode(multiArray)
    }

    // MARK: - Utility Methods

    /// Get information about the loaded model
    var modelDescription: String {
        let desc = model.modelDescription
        var info = "VoxtralCoreMLEncoder:\n"
        info += "  Input: \(desc.inputDescriptionsByName.keys.joined(separator: ", "))\n"
        info += "  Output: \(desc.outputDescriptionsByName.keys.joined(separator: ", "))\n"
        if let metadata = desc.metadata[.description] as? String {
            info += "  Description: \(metadata)\n"
        }
        return info
    }

    /// Check if Core ML with ANE is available on this device
    static var isANEAvailable: Bool {
        // ANE is available on Apple Silicon Macs and A11+ iOS devices
        #if os(macOS)
        // Check for Apple Silicon
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingCString: $0)
            }
        }
        return machine?.contains("arm64") ?? false
        #else
        // iOS devices with A11 or later have ANE
        return true
        #endif
    }

    // MARK: - HuggingFace Download

    /// Default HuggingFace repository for Core ML encoder (Mini variant)
    static let defaultHuggingFaceRepo = "VincentGOURBIN/voxtral-encoder-coreml-mini"

    /// Default model name in the repository
    static let defaultModelName = "VoxtralEncoderMini.mlmodelc"

    /// Download Core ML encoder from HuggingFace Hub
    /// - Parameters:
    ///   - variant: Model variant (mini or small) - determines repo and model name
    ///   - progress: Optional progress callback (progress 0-1, status message)
    /// - Returns: URL to the downloaded model
    public static func downloadFromHuggingFace(
        variant: VoxtralCoreMLVariant = .mini,
        progress: ((Double, String) -> Void)? = nil
    ) async throws -> URL {
        let repo = variant.huggingFaceRepo
        let modelName = variant.modelName

        progress?(0.0, "Checking cache for \(variant.rawValue) encoder...")

        // Same layout and completeness manifest as every other model (K-6):
        // modelsDirectory/<org>/<repo>/<name>, under customModelsDirectory when the app sets one.
        // No HubApi: nothing is written to ~/.cache/huggingface and a verified copy loads offline.
        let repoDir = VoxtralModelDownloader.modelsDirectory.appendingPathComponent(repo)
        let modelPath = repoDir.appendingPathComponent(modelName)
        if isCompleteEncoder(repoDir: repoDir, modelPath: modelPath) {
            progress?(1.0, "Core ML \(variant.rawValue) model found in cache")
            return modelPath
        }

        progress?(0.1, "Downloading \(variant.rawValue) encoder from HuggingFace...")
        VoxtralDebug.log("Downloading Core ML \(variant.rawValue) encoder from \(repo)")

        let callback = Locked(progress)
        do {
            // The glob does not cross "/" (no `**`): the .mlmodelc is two levels deep at most
            _ = try await VoxtralModelDownloader.downloadRepoDirect(
                repoId: repo,
                matching: ["\(modelName)/*", "\(modelName)/*/*"],
                progress: { fraction, message in callback.get()?(0.1 + 0.85 * fraction, message) }
            )
        } catch {
            throw VoxtralCoreMLError.modelLoadFailed("Failed to download \(variant.rawValue) encoder from HuggingFace: \(error.localizedDescription)")
        }

        guard isCompleteEncoder(repoDir: repoDir, modelPath: modelPath) else {
            throw VoxtralCoreMLError.modelNotFound("\(modelName) incomplete after download from \(repo) (\(repoDir.path))")
        }
        progress?(1.0, "Core ML \(variant.rawValue) encoder downloaded!")
        return modelPath
    }

    /// A downloaded encoder is usable when its repo folder carries a verified manifest
    /// and the compiled model holds both its program and its weights.
    private static func isCompleteEncoder(repoDir: URL, modelPath: URL) -> Bool {
        VoxtralModelDownloader.hasCompleteManifest(repoDir)
            && VoxtralModelDownloader.fileSize(at: modelPath.appendingPathComponent("model.mil")) != nil
            && VoxtralModelDownloader.fileSize(at: modelPath.appendingPathComponent("weights/weight.bin")) != nil
    }

    /// Download Core ML encoder for a specific MLX model
    /// Automatically selects the correct variant based on the MLX model's configuration
    /// - Parameters:
    ///   - mlxModelRepoId: The MLX model repository ID to match
    ///   - progress: Optional progress callback
    /// - Returns: URL to the downloaded model
    public static func downloadForMLXModel(
        mlxModelRepoId: String,
        progress: ((Double, String) -> Void)? = nil
    ) async throws -> URL {
        let variant = VoxtralCoreMLVariant.fromMLXModelRepoId(mlxModelRepoId)
        VoxtralDebug.log("Selected Core ML variant '\(variant.rawValue)' for MLX model: \(mlxModelRepoId)")
        return try await downloadFromHuggingFace(variant: variant, progress: progress)
    }

    /// Convenience initializer that downloads from HuggingFace if needed
    /// - Parameters:
    ///   - variant: Model variant to download
    ///   - config: Core ML configuration (will be updated with variant if needed)
    ///   - progress: Optional progress callback
    static func fromHuggingFace(
        variant: VoxtralCoreMLVariant = .mini,
        config: VoxtralCoreMLConfig? = nil,
        progress: ((Double, String) -> Void)? = nil
    ) async throws -> VoxtralCoreMLEncoder {
        let modelURL = try await downloadFromHuggingFace(variant: variant, progress: progress)

        // Use provided config or create one with the correct variant
        var finalConfig = config ?? .default
        finalConfig.variant = variant

        return try VoxtralCoreMLEncoder(modelURL: modelURL, config: finalConfig)
    }

    /// Convenience initializer that auto-selects variant based on MLX model
    /// - Parameters:
    ///   - mlxModelRepoId: MLX model repository ID to match variant
    ///   - config: Core ML configuration
    ///   - progress: Optional progress callback
    static func forMLXModel(
        mlxModelRepoId: String,
        config: VoxtralCoreMLConfig? = nil,
        progress: ((Double, String) -> Void)? = nil
    ) async throws -> VoxtralCoreMLEncoder {
        let variant = VoxtralCoreMLVariant.fromMLXModelRepoId(mlxModelRepoId)
        return try await fromHuggingFace(variant: variant, config: config, progress: progress)
    }
}

// MARK: - Async Support

extension VoxtralCoreMLEncoder {

    /// Async version of encode
    /// - Parameter melSpectrogram: Mel spectrogram as MLMultiArray
    /// - Returns: Audio embeddings as MLMultiArray
    func encodeAsync(_ melSpectrogram: MLMultiArray) async throws -> MLMultiArray {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try self.encode(melSpectrogram)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

// MARK: - Performance Monitoring

extension VoxtralCoreMLEncoder {

    /// Measure encoding time for benchmarking
    /// - Parameter melSpectrogram: Input mel spectrogram
    /// - Returns: Tuple of (result, timeInMilliseconds)
    func encodeWithTiming(_ melSpectrogram: MLMultiArray) throws -> (MLMultiArray, Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try encode(melSpectrogram)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        return (result, elapsed)
    }

    /// Run benchmark with multiple iterations
    /// - Parameters:
    ///   - iterations: Number of iterations
    ///   - warmup: Number of warmup iterations
    /// - Returns: Average time in milliseconds
    func benchmark(iterations: Int = 10, warmup: Int = 3) throws -> Double {
        // Create test input
        let testInput = try MLMultiArray(
            shape: config.inputShape.map { NSNumber(value: $0) },
            dataType: .float32
        )

        // Warmup
        VoxtralDebug.log("Core ML benchmark: \(warmup) warmup iterations...")
        for _ in 0..<warmup {
            _ = try encode(testInput)
        }

        // Benchmark
        VoxtralDebug.log("Core ML benchmark: \(iterations) timed iterations...")
        var totalTime: Double = 0
        for _ in 0..<iterations {
            let (_, time) = try encodeWithTiming(testInput)
            totalTime += time
        }

        let avgTime = totalTime / Double(iterations)
        VoxtralDebug.log("Core ML benchmark: avg \(String(format: "%.2f", avgTime))ms per inference")
        return avgTime
    }
}
