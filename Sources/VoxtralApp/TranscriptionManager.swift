/**
 * TranscriptionManager - Handles model loading and audio processing for the app
 * Supports both transcription and chat modes with real-time streaming
 * Now with model selection and automatic downloading
 */

import Foundation
import SwiftUI
import VoxtralCore
import MLX

@MainActor
class TranscriptionManager: ObservableObject {
    // Model state
    @Published var isLoading = false
    @Published var isModelLoaded = false
    @Published var errorMessage: String?
    @Published var loadingStatus: String = ""

    // Download progress
    @Published var isDownloading = false
    @Published var downloadProgress: Double = 0.0
    @Published var downloadMessage: String = ""

    // Model selection
    @Published var selectedModelId: String = "mini-3b-8bit"
    @Published var availableModels: [VoxtralModelInfo] = ModelRegistry.models
    @Published var downloadedModels: Set<String> = []
    @Published var modelSizes: [String: Int64] = [:]

    // Processing state
    @Published var isTranscribing = false
    @Published var transcription = ""
    @Published var lastGenerationStats: GenerationStats?
    @Published var currentTokenCount = 0
    @Published var currentStep: ProcessingStep = .idle

    enum ProcessingStep: String {
        case idle = ""
        case processingAudio = "Processing audio..."
        case settingUpGeneration = "Setting up generation..."
        case generating = "Generating"

        var icon: String {
            switch self {
            case .idle: return ""
            case .processingAudio: return "waveform"
            case .settingUpGeneration: return "gearshape.2"
            case .generating: return "text.cursor"
            }
        }
    }

    // Profiling
    @Published var lastProfileSummary: ProfileSummary?
    @Published var profilingEnabled = true  // Enable profiling by default
    private var profiler = VoxtralProfiler()

    // Mode and settings
    @Published var mode: VoxtralMode = .transcription
    @Published var selectedAudioPath: String?
    @Published var chatPrompt: String = "What is being said in this audio?"
    @Published var maxTokens: Int = 500
    @Published var temperature: Float = 0.0
    @Published var contextSize: Int = 8192  // KV cache size limit (1024-32768), default 8k balanced
    @Published var useHybridBackend: Bool = false  // Core ML encoder + MLX decoder

    // Hybrid encoder status (updated after model load)
    @Published var hybridEncoderAvailable: Bool = false

    // Private model references
    private var model: VoxtralForConditionalGeneration?
    private var standardModel: VoxtralStandardModel?
    private var processor: VoxtralProcessor?
    private var hybridEncoder: VoxtralHybridEncoder?
    @Published private(set) var currentLoadedModelId: String?

    var canRun: Bool {
        isModelLoaded && selectedAudioPath != nil && !isTranscribing
    }

    var selectedModel: VoxtralModelInfo? {
        ModelRegistry.model(withId: selectedModelId)
    }

    var isCurrentModelLoaded: Bool {
        isModelLoaded && currentLoadedModelId == selectedModelId
    }

    init() {
        refreshDownloadedModels()
    }

    // MARK: - Model Management

    func refreshDownloadedModels() {
        let downloaded = ModelDownloader.listDownloadedModels()
        downloadedModels = Set(downloaded.map { $0.id })

        // Also refresh model sizes
        var sizes: [String: Int64] = [:]
        for model in downloaded {
            if let size = ModelDownloader.modelSize(for: model) {
                sizes[model.id] = size
            }
        }
        modelSizes = sizes
    }

    func isModelDownloaded(_ modelId: String) -> Bool {
        if let model = ModelRegistry.model(withId: modelId) {
            return ModelDownloader.findModelPath(for: model) != nil
        }
        return false
    }

    func deleteModel(_ modelId: String) async throws {
        guard let model = ModelRegistry.model(withId: modelId) else { return }

        // Unload if currently loaded
        if currentLoadedModelId == modelId {
            unloadModel()
        }

        try ModelDownloader.deleteModel(model)
        refreshDownloadedModels()
    }

    func downloadModel(_ modelId: String) async {
        guard let model = ModelRegistry.model(withId: modelId) else { return }
        guard !isDownloading else { return }

        isDownloading = true
        downloadProgress = 0.0
        downloadMessage = "Starting download..."
        errorMessage = nil

        do {
            _ = try await ModelDownloader.download(model) { @Sendable [weak self] progress, message in
                Task { @MainActor in
                    self?.downloadProgress = progress
                    self?.downloadMessage = message
                }
            }
            downloadMessage = "Download complete!"
            refreshDownloadedModels()
        } catch {
            errorMessage = "Download failed: \(error.localizedDescription)"
        }

        isDownloading = false
    }

    // MARK: - Model Loading

    func loadModel() async {
        guard !isLoading else {
            print("[VoxtralApp] loadModel: already loading, skipping")
            return
        }

        // If same model already loaded, skip
        if isModelLoaded && currentLoadedModelId == selectedModelId {
            print("[VoxtralApp] loadModel: model \(selectedModelId) already loaded")
            return
        }

        // Unload previous model to free memory before loading new one
        if isModelLoaded {
            print("[VoxtralApp] Unloading previous model to free memory...")
            unloadModel()
        }

        print("[VoxtralApp] loadModel: starting to load \(selectedModelId)")
        isLoading = true
        isModelLoaded = false
        errorMessage = nil
        loadingStatus = "Resolving model..."

        do {
            // Resolve model path (downloads if needed)
            print("[VoxtralApp] Resolving model path...")
            loadingStatus = "Checking model..."

            // Check if download is needed
            let needsDownload = !isModelDownloaded(selectedModelId)
            if needsDownload {
                isDownloading = true
                downloadProgress = 0.0
            }

            let modelPath = try await ModelDownloader.resolveModel(selectedModelId) { @Sendable [weak self] progress, message in
                Task { @MainActor in
                    self?.loadingStatus = message
                    if needsDownload {
                        self?.downloadProgress = progress
                        self?.downloadMessage = message
                    }
                }
            }

            isDownloading = false
            print("[VoxtralApp] Model path resolved: \(modelPath.path)")

            loadingStatus = "Loading model weights..."
            print("[VoxtralApp] Loading model weights...")

            let path = modelPath.path
            let (loadedModel, _) = try await Task.detached(priority: .userInitiated) {
                try loadVoxtralStandardModel(modelPath: path, dtype: .float16)
            }.value
            print("[VoxtralApp] Model weights loaded")

            loadingStatus = "Initializing processor..."
            print("[VoxtralApp] Initializing processor...")

            let wrapper = VoxtralForConditionalGeneration(standardModel: loadedModel)
            let loadedProcessor = try VoxtralProcessor.fromPretrained(path)

            self.standardModel = loadedModel
            self.model = wrapper
            self.processor = loadedProcessor
            self.isModelLoaded = true
            self.currentLoadedModelId = selectedModelId
            self.loadingStatus = ""

            // Initialize hybrid encoder if available (Core ML + MLX)
            if #available(macOS 13.0, iOS 16.0, *) {
                self.hybridEncoder = wrapper.createHybridEncoder(preferredBackend: .auto)
                self.hybridEncoderAvailable = self.hybridEncoder?.status.coreMLAvailable ?? false
                if self.hybridEncoderAvailable {
                    print("[VoxtralApp] Hybrid encoder initialized (Core ML available)")
                } else {
                    print("[VoxtralApp] Hybrid encoder: Core ML not available, will use MLX only")
                }
            } else {
                self.hybridEncoderAvailable = false
            }

            print("[VoxtralApp] Model loaded successfully!")

            // Refresh downloaded models list
            refreshDownloadedModels()

        } catch {
            print("[VoxtralApp] Error loading model: \(error)")
            self.errorMessage = error.localizedDescription
            self.loadingStatus = ""
        }

        isLoading = false
    }

    func unloadModel() {
        model = nil
        standardModel = nil
        processor = nil
        hybridEncoder = nil
        hybridEncoderAvailable = false
        isModelLoaded = false
        currentLoadedModelId = nil

        // Clear GPU cache to release memory
        Memory.clearCache()
        GPU.resetPeakMemory()
    }

    // MARK: - Memory Management

    /// Clear MLX GPU cache to free memory
    func clearCache() {
        Memory.clearCache()
    }

    /// Aggressive memory cleanup - clears cache and resets peak tracking
    func aggressiveMemoryCleanup() {
        // Clear the recyclable cache
        Memory.clearCache()
        // Reset peak memory tracking
        GPU.resetPeakMemory()
        // Try setting a temporary low cache limit to force cleanup
        let currentCache = Memory.cacheMemory
        if currentCache > 0 {
            Memory.cacheLimit = 0  // Temporarily disable caching
            Memory.clearCache()
            Memory.cacheLimit = Int.max  // Restore default (unlimited)
        }
    }

    /// Get current MLX memory stats
    var memoryStats: (active: Int, cache: Int, peak: Int) {
        (Memory.activeMemory, Memory.cacheMemory, Memory.peakMemory)
    }

    /// Format bytes as human-readable string
    static func formatBytes(_ bytes: Int) -> String {
        let absBytes = abs(bytes)
        if absBytes >= 1024 * 1024 * 1024 {
            return String(format: "%.2f GB", Double(bytes) / (1024 * 1024 * 1024))
        } else if absBytes >= 1024 * 1024 {
            return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
        } else if absBytes >= 1024 {
            return String(format: "%.1f KB", Double(bytes) / 1024)
        }
        return "\(bytes) B"
    }

    // MARK: - Run (dispatch to appropriate mode)

    func run() async {
        switch mode {
        case .transcription:
            await transcribe()
        case .chat:
            await chat()
        }
    }

    // MARK: - Transcription Mode

    func transcribe() async {
        guard let model = model,
              let processor = processor,
              let audioPath = selectedAudioPath else { return }

        isTranscribing = true
        transcription = ""
        lastGenerationStats = nil
        lastProfileSummary = nil
        currentTokenCount = 0
        currentStep = .processingAudio

        // Start profiling
        if profilingEnabled {
            profiler.start()
        }

        let startTime = Date()

        do {
            // Step 1: Process audio
            currentStep = .processingAudio
            await Task.yield()  // Allow UI to update

            let inputs: ProcessedInputs
            if profilingEnabled {
                inputs = try profiler.profile("Audio Processing") {
                    try processor.applyTranscritionRequest(
                        audio: audioPath,
                        language: "en",
                        samplingRate: 16000
                    )
                }
            } else {
                inputs = try processor.applyTranscritionRequest(
                    audio: audioPath,
                    language: "en",
                    samplingRate: 16000
                )
            }

            // Step 2: Setup generation
            currentStep = .settingUpGeneration
            await Task.yield()  // Allow UI to update

            // 🚀 generateStream now returns [Int] directly - no GPU references to clean
            // Use hybrid encoder (Core ML + MLX) if available and enabled
            var tokenIds: [Int]
            if #available(macOS 13.0, iOS 16.0, *), useHybridBackend, let hybrid = hybridEncoder {
                // Hybrid path: Core ML encoder + MLX decoder
                if profilingEnabled {
                    tokenIds = try profiler.profile("Generation (Hybrid)") {
                        let audioEmbeds = try hybrid.encode(inputs.inputFeatures)
                        return try model.generateStreamWithAudioEmbeds(
                            inputIds: inputs.inputIds,
                            audioEmbeds: audioEmbeds,
                            attentionMask: nil,
                            maxNewTokens: maxTokens,
                            temperature: temperature,
                            topP: 1.0,
                            repetitionPenalty: 1.1,
                            contextSize: contextSize
                        )
                    }
                } else {
                    let audioEmbeds = try hybrid.encode(inputs.inputFeatures)
                    tokenIds = try model.generateStreamWithAudioEmbeds(
                        inputIds: inputs.inputIds,
                        audioEmbeds: audioEmbeds,
                        attentionMask: nil,
                        maxNewTokens: maxTokens,
                        temperature: temperature,
                        topP: 1.0,
                        repetitionPenalty: 1.1,
                        contextSize: contextSize
                    )
                }
            } else {
                // Full MLX path
                if profilingEnabled {
                    tokenIds = try profiler.profile("Generation") {
                        try model.generateStream(
                            inputIds: inputs.inputIds,
                            inputFeatures: inputs.inputFeatures,
                            attentionMask: nil,
                            maxNewTokens: maxTokens,
                            temperature: temperature,
                            topP: 1.0,
                            repetitionPenalty: 1.1,
                            contextSize: contextSize
                        )
                    }
                } else {
                    tokenIds = try model.generateStream(
                        inputIds: inputs.inputIds,
                        inputFeatures: inputs.inputFeatures,
                        attentionMask: nil,
                        maxNewTokens: maxTokens,
                        temperature: temperature,
                        topP: 1.0,
                        repetitionPenalty: 1.1,
                        contextSize: contextSize
                    )
                }
            }

            // Step 3: Decode tokens to text
            currentStep = .generating
            currentTokenCount = tokenIds.count

            // Decode all tokens at once
            if let text = try? processor.decode(tokenIds) {
                transcription = text
            }

            let duration = Date().timeIntervalSince(startTime)
            lastGenerationStats = GenerationStats(tokenCount: currentTokenCount, duration: duration)

            // Get profile summary
            if profilingEnabled {
                lastProfileSummary = profiler.summary()
                print(lastProfileSummary!.description)
            }

        } catch {
            transcription = "Error: \(error.localizedDescription)"
        }

        currentStep = .idle
        isTranscribing = false

        // 🧹 Release GPU memory after generation
        Memory.clearCache()
    }

    // MARK: - Chat Mode

    func chat() async {
        guard let model = model,
              let processor = processor,
              let audioPath = selectedAudioPath else { return }

        isTranscribing = true
        transcription = ""
        lastGenerationStats = nil
        lastProfileSummary = nil
        currentTokenCount = 0
        currentStep = .processingAudio

        // Start profiling
        if profilingEnabled {
            profiler.start()
        }

        let startTime = Date()

        do {
            // Step 1: Process audio and build chat template
            currentStep = .processingAudio
            await Task.yield()

            // Build chat conversation with audio and prompt
            let conversation: [[String: Any]] = [
                [
                    "role": "user",
                    "content": [
                        ["type": "audio", "audio": audioPath],
                        ["type": "text", "text": chatPrompt]
                    ]
                ]
            ]

            let inputs: ProcessedInputs
            if profilingEnabled {
                inputs = try profiler.profile("Chat Template") {
                    let chatResult = try processor.applyChatTemplate(
                        conversation: conversation,
                        tokenize: true,
                        returnTensors: "mlx"
                    ) as! [String: MLXArray]

                    return ProcessedInputs(
                        inputIds: chatResult["input_ids"]!,
                        inputFeatures: chatResult["input_features"]!
                    )
                }
            } else {
                let chatResult = try processor.applyChatTemplate(
                    conversation: conversation,
                    tokenize: true,
                    returnTensors: "mlx"
                ) as! [String: MLXArray]

                inputs = ProcessedInputs(
                    inputIds: chatResult["input_ids"]!,
                    inputFeatures: chatResult["input_features"]!
                )
            }

            // Step 2: Setup generation
            currentStep = .settingUpGeneration
            await Task.yield()

            // 🚀 generateStream now returns [Int] directly - no GPU references to clean
            // Use hybrid encoder (Core ML + MLX) if available and enabled
            var tokenIds: [Int]
            if #available(macOS 13.0, iOS 16.0, *), useHybridBackend, let hybrid = hybridEncoder {
                // Hybrid path: Core ML encoder + MLX decoder
                if profilingEnabled {
                    tokenIds = try profiler.profile("Generation (Hybrid)") {
                        let audioEmbeds = try hybrid.encode(inputs.inputFeatures)
                        return try model.generateStreamWithAudioEmbeds(
                            inputIds: inputs.inputIds,
                            audioEmbeds: audioEmbeds,
                            attentionMask: nil,
                            maxNewTokens: maxTokens,
                            temperature: temperature,
                            topP: 1.0,
                            repetitionPenalty: 1.1,
                            contextSize: contextSize
                        )
                    }
                } else {
                    let audioEmbeds = try hybrid.encode(inputs.inputFeatures)
                    tokenIds = try model.generateStreamWithAudioEmbeds(
                        inputIds: inputs.inputIds,
                        audioEmbeds: audioEmbeds,
                        attentionMask: nil,
                        maxNewTokens: maxTokens,
                        temperature: temperature,
                        topP: 1.0,
                        repetitionPenalty: 1.1,
                        contextSize: contextSize
                    )
                }
            } else {
                // Full MLX path
                if profilingEnabled {
                    tokenIds = try profiler.profile("Generation") {
                        try model.generateStream(
                            inputIds: inputs.inputIds,
                            inputFeatures: inputs.inputFeatures,
                            attentionMask: nil,
                            maxNewTokens: maxTokens,
                            temperature: temperature,
                            topP: 1.0,
                            repetitionPenalty: 1.1,
                            contextSize: contextSize
                        )
                    }
                } else {
                    tokenIds = try model.generateStream(
                        inputIds: inputs.inputIds,
                        inputFeatures: inputs.inputFeatures,
                        attentionMask: nil,
                        maxNewTokens: maxTokens,
                        temperature: temperature,
                        topP: 1.0,
                        repetitionPenalty: 1.1,
                        contextSize: contextSize
                    )
                }
            }

            // Step 3: Decode tokens to text
            currentStep = .generating
            currentTokenCount = tokenIds.count

            // Decode all tokens at once
            if let text = try? processor.decode(tokenIds) {
                transcription = text
            }

            let duration = Date().timeIntervalSince(startTime)
            lastGenerationStats = GenerationStats(tokenCount: currentTokenCount, duration: duration)

            // Get profile summary
            if profilingEnabled {
                lastProfileSummary = profiler.summary()
                print(lastProfileSummary!.description)
            }

        } catch {
            transcription = "Error: \(error.localizedDescription)"
        }

        currentStep = .idle
        isTranscribing = false

        // 🧹 Release GPU memory after generation
        Memory.clearCache()
    }
}
