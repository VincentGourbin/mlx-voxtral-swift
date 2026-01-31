/**
 * TranscriptionManager - Handles model loading and audio processing for the app
 * 
 * IMPORTANT: This manager uses ONLY the high-level VoxtralPipeline API.
 * This ensures the public API works correctly and is properly tested.
 *
 * Supports both transcription and chat modes with real-time streaming.
 * Includes model selection and automatic downloading.
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

    // High-level Pipeline (replaces low-level model references)
    private var pipeline: VoxtralPipeline?
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

    // MARK: - Model Loading (Uses VoxtralPipeline API)

    /// Convert model ID to VoxtralPipeline.Model enum
    private func pipelineModel(for modelId: String) -> VoxtralPipeline.Model? {
        switch modelId.lowercased() {
        case "mini-3b", "mini3b": return .mini3b
        case "mini-3b-8bit", "mini3b8bit": return .mini3b8bit
        case "mini-3b-4bit", "mini3b4bit": return .mini3b4bit
        case "small-24b", "small24b": return .small24b
        case "small-24b-8bit", "small24b8bit": return .small24b8bit
        case "small-4bit", "small4bit": return .small4bit
        default: return nil
        }
    }

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
            // Get pipeline model enum
            guard let pipelineModelEnum = pipelineModel(for: selectedModelId) else {
                throw NSError(domain: "VoxtralApp", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unknown model: \(selectedModelId)"])
            }

            // Check if download is needed
            let needsDownload = !isModelDownloaded(selectedModelId)
            if needsDownload {
                isDownloading = true
                downloadProgress = 0.0
            }

            // Create pipeline configuration
            var config = VoxtralPipeline.Configuration.default
            config.maxTokens = maxTokens
            config.temperature = temperature
            config.memoryOptimization.maxKVCacheSize = contextSize

            // Always use .auto to download Core ML if available
            // The useHybridBackend toggle controls whether we USE Core ML, not whether we download it
            let newPipeline = VoxtralPipeline(
                model: pipelineModelEnum,
                backend: .auto,  // Always try to get Core ML
                configuration: config
            )

            // Load model using high-level API
            try await newPipeline.loadModel { @Sendable [weak self] progress, message in
                Task { @MainActor in
                    self?.loadingStatus = message
                    if needsDownload {
                        self?.downloadProgress = progress
                        self?.downloadMessage = message
                    }
                }
            }

            isDownloading = false
            self.pipeline = newPipeline
            self.isModelLoaded = true
            self.currentLoadedModelId = selectedModelId
            self.loadingStatus = ""

            // Check hybrid encoder availability from pipeline status
            let encoderStatus = newPipeline.encoderStatus
            print("[VoxtralApp] Encoder status: \(encoderStatus)")
            self.hybridEncoderAvailable = encoderStatus.contains("Core ML available: true")
            
            if self.hybridEncoderAvailable {
                print("[VoxtralApp] ✅ Hybrid encoder initialized (Core ML available)")
            } else {
                print("[VoxtralApp] ⚠️ Core ML not available, encoderStatus=\(encoderStatus)")
            }

            print("[VoxtralApp] Model loaded successfully via VoxtralPipeline!")

            // Refresh downloaded models list
            refreshDownloadedModels()

        } catch {
            print("[VoxtralApp] Error loading model: \(error)")
            self.errorMessage = error.localizedDescription
            self.loadingStatus = ""
            isDownloading = false
        }

        isLoading = false
    }

    func unloadModel() {
        pipeline?.unload()
        pipeline = nil
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

    // MARK: - Transcription Mode (Uses VoxtralPipeline API)

    func transcribe() async {
        guard let pipeline = pipeline,
              let audioPath = selectedAudioPath else { return }

        isTranscribing = true
        transcription = ""
        lastGenerationStats = nil
        lastProfileSummary = nil
        currentTokenCount = 0
        currentStep = .processingAudio

        // Update pipeline configuration if needed
        pipeline.configuration.maxTokens = maxTokens
        pipeline.configuration.temperature = temperature
        pipeline.configuration.memoryOptimization.maxKVCacheSize = contextSize

        // Start profiling
        if profilingEnabled {
            profiler.start()
        }

        let startTime = Date()

        do {
            currentStep = .settingUpGeneration
            await Task.yield()  // Allow UI to update

            let audioURL = URL(fileURLWithPath: audioPath)
            
            // Use high-level pipeline API
            let result = try await pipeline.transcribe(audio: audioURL, language: "en")

            currentStep = .generating
            transcription = result
            currentTokenCount = result.split(separator: " ").count  // Approximate

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

    // MARK: - Chat Mode (Uses VoxtralPipeline API)

    func chat() async {
        guard let pipeline = pipeline,
              let audioPath = selectedAudioPath else { return }

        isTranscribing = true
        transcription = ""
        lastGenerationStats = nil
        lastProfileSummary = nil
        currentTokenCount = 0
        currentStep = .processingAudio

        // Update pipeline configuration if needed
        pipeline.configuration.maxTokens = maxTokens
        pipeline.configuration.temperature = temperature
        pipeline.configuration.memoryOptimization.maxKVCacheSize = contextSize

        // Start profiling
        if profilingEnabled {
            profiler.start()
        }

        let startTime = Date()

        do {
            currentStep = .settingUpGeneration
            await Task.yield()

            let audioURL = URL(fileURLWithPath: audioPath)

            // Use high-level pipeline API
            let result = try await pipeline.chat(audio: audioURL, prompt: chatPrompt, language: "en")

            currentStep = .generating
            transcription = result
            currentTokenCount = result.split(separator: " ").count  // Approximate

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
