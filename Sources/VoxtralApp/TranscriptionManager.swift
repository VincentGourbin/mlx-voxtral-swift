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

    // Mode and settings
    @Published var mode: VoxtralMode = .transcription
    @Published var selectedAudioPath: String?
    @Published var chatPrompt: String = "What is being said in this audio?"
    @Published var maxTokens: Int = 500
    @Published var temperature: Float = 0.0

    // Private model references
    private var model: VoxtralForConditionalGeneration?
    private var standardModel: VoxtralStandardModel?
    private var processor: VoxtralProcessor?
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
            _ = try await ModelDownloader.download(model) { progress, message in
                Task { @MainActor in
                    self.downloadProgress = progress
                    self.downloadMessage = message
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

            let modelPath = try await ModelDownloader.resolveModel(selectedModelId) { progress, message in
                Task { @MainActor in
                    self.loadingStatus = message
                    if needsDownload {
                        self.downloadProgress = progress
                        self.downloadMessage = message
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
        isModelLoaded = false
        currentLoadedModelId = nil
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
        currentTokenCount = 0

        let startTime = Date()

        do {
            let inputs = try processor.applyTranscritionRequest(
                audio: audioPath,
                language: "en",
                samplingRate: 16000
            )

            let streamResults = try model.generateStream(
                inputIds: inputs.inputIds,
                inputFeatures: inputs.inputFeatures,
                attentionMask: nil,
                maxNewTokens: maxTokens,
                temperature: temperature,
                topP: 1.0,
                repetitionPenalty: 1.1
            )

            for (token, _) in streamResults {
                let tokenId = token.item(Int.self)
                currentTokenCount += 1

                if let tokenText = try? processor.decode([tokenId]) {
                    transcription += tokenText
                }

                // Yield to allow UI to update - this is the key for streaming!
                await Task.yield()
            }

            let duration = Date().timeIntervalSince(startTime)
            lastGenerationStats = GenerationStats(tokenCount: currentTokenCount, duration: duration)

        } catch {
            transcription = "Error: \(error.localizedDescription)"
        }

        isTranscribing = false
    }

    // MARK: - Chat Mode

    func chat() async {
        guard let model = model,
              let processor = processor,
              let audioPath = selectedAudioPath else { return }

        isTranscribing = true
        transcription = ""
        lastGenerationStats = nil
        currentTokenCount = 0

        let startTime = Date()

        do {
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

            let chatResult = try processor.applyChatTemplate(
                conversation: conversation,
                tokenize: true,
                returnTensors: "mlx"
            ) as! [String: MLXArray]

            let inputs = ProcessedInputs(
                inputIds: chatResult["input_ids"]!,
                inputFeatures: chatResult["input_features"]!
            )

            let streamResults = try model.generateStream(
                inputIds: inputs.inputIds,
                inputFeatures: inputs.inputFeatures,
                attentionMask: nil,
                maxNewTokens: maxTokens,
                temperature: temperature,
                topP: 1.0,
                repetitionPenalty: 1.1
            )

            for (token, _) in streamResults {
                let tokenId = token.item(Int.self)
                currentTokenCount += 1

                if let tokenText = try? processor.decode([tokenId]) {
                    transcription += tokenText
                }

                // Yield to allow UI to update - this is the key for streaming!
                await Task.yield()
            }

            let duration = Date().timeIntervalSince(startTime)
            lastGenerationStats = GenerationStats(tokenCount: currentTokenCount, duration: duration)

        } catch {
            transcription = "Error: \(error.localizedDescription)"
        }

        isTranscribing = false
    }
}
