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

    // Model selection
    @Published var selectedModelId: String = "mini-3b-8bit"
    @Published var availableModels: [VoxtralModelInfo] = ModelRegistry.models
    @Published var downloadedModels: Set<String> = []

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
    private var currentLoadedModelId: String?

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
    }

    func isModelDownloaded(_ modelId: String) -> Bool {
        if let model = ModelRegistry.model(withId: modelId) {
            return ModelDownloader.findModelPath(for: model) != nil
        }
        return false
    }

    // MARK: - Model Loading

    func loadModel() async {
        guard !isLoading else { return }

        // If same model already loaded, skip
        if isModelLoaded && currentLoadedModelId == selectedModelId {
            return
        }

        isLoading = true
        isModelLoaded = false
        errorMessage = nil
        loadingStatus = "Resolving model..."

        do {
            // Resolve model path (downloads if needed)
            loadingStatus = "Downloading model if needed..."
            let modelPath = try await ModelDownloader.resolveModel(selectedModelId) { progress, message in
                Task { @MainActor in
                    self.loadingStatus = message
                }
            }

            loadingStatus = "Loading model weights..."

            let path = modelPath.path
            let (loadedModel, _) = try await Task.detached(priority: .userInitiated) {
                try loadVoxtralStandardModel(modelPath: path, dtype: .float16)
            }.value

            loadingStatus = "Initializing processor..."

            let wrapper = VoxtralForConditionalGeneration(standardModel: loadedModel)
            let loadedProcessor = try VoxtralProcessor.fromPretrained(path)

            self.standardModel = loadedModel
            self.model = wrapper
            self.processor = loadedProcessor
            self.isModelLoaded = true
            self.currentLoadedModelId = selectedModelId
            self.loadingStatus = ""

            // Refresh downloaded models list
            refreshDownloadedModels()

        } catch {
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
