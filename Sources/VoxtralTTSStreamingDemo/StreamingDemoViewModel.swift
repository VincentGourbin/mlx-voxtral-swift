import Foundation
import AVFoundation
import VoxtralCore
import MLX

@available(macOS 14.0, *)
@MainActor
final class StreamingDemoViewModel: ObservableObject {

    // MARK: - User inputs

    @Published var text: String = "Fluxforge Studio transforme votre Mac en un studio de création IA complet. Générez des images et des vidéos de haute qualité à partir de texte, entraînez vos propres modèles personnalisés, et gérez votre bibliothèque créative — le tout en local sur votre Apple Silicon, sans cloud ni abonnement."
    @Published var selectedModelId: String = "tts-4b-4bit"
    @Published var selectedVoice: String = "fr_female"
    @Published var sanitizeEnabled: Bool = true

    // MARK: - State

    @Published var isModelLoaded = false
    @Published var isLoading = false
    @Published var isSynthesizing = false
    @Published var loadProgress: Double = 0
    @Published var loadStatus: String = ""

    // MARK: - Metrics

    @Published var buttonClickTime: Date?
    @Published var ttft: TimeInterval?
    @Published var totalTime: TimeInterval?
    @Published var audioDuration: TimeInterval = 0
    @Published var rtf: Double?
    @Published var framesGenerated: Int = 0
    @Published var chunksReceived: Int = 0
    @Published var currentModelName: String = ""
    @Published var fps: Double = 0

    // MARK: - Log

    @Published var logLines: [String] = []

    // MARK: - Private

    private var pipeline: VoxtralTTSPipeline?
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var streamingTask: Task<Void, Never>?
    private var currentModelId: String?

    let availableModels: [(id: String, name: String)] = [
        ("tts-4b-4bit", "4-bit (2.5 GB)"),
        ("tts-4b-6bit", "6-bit (3.5 GB)"),
        ("tts-4b-mlx", "bf16 (8 GB)")
    ]

    let availableVoices: [(id: String, label: String)] = VoxtralVoice.allCases.map {
        ($0.rawValue, $0.displayName)
    }

    struct TextPreset {
        let label: String
        let text: String
    }

    let textPresets: [TextPreset] = [
        TextPreset(label: "Short FR", text: "Fluxforge Studio transforme votre Mac en un studio de création IA complet."),
        TextPreset(label: "Short EN", text: "Fluxforge Studio turns your Mac into a complete AI creative studio."),
        TextPreset(label: "Long FR", text: """
Fluxforge Studio transforme votre Mac en un studio de création IA complet. Générez des images et des vidéos de haute qualité à partir de texte, entraînez vos propres modèles personnalisés, et gérez votre bibliothèque créative — le tout en local sur votre Apple Silicon, sans cloud ni abonnement.

FORGE TON IDÉE
Un atelier créatif complet pour explorer vos idées visuelles. Décrivez votre concept en texte, importez une image ou un audio, puis itérez librement : variations, changements de style, animations vidéo. Chaque étape est sauvegardée dans un arbre de branches façon Git — rien ne se perd, tout se retrouve.

GÉNÉRATION D'IMAGES AVANCÉE
Quatre modèles Flux 2 au choix selon vos besoins. Ajustez la résolution, les étapes d'inférence, le guidance, utilisez des images de référence, et activez l'amélioration automatique du prompt.

GÉNÉRATION VIDÉO
Créez des vidéos à partir de texte ou d'images grâce au modèle LTX-2.3. Deux variantes : Distilled (rapide) et Dev (haute qualité). Ajoutez une bande-son générée automatiquement.

100% LOCAL ET PRIVÉ
Aucun compte requis. Aucune donnée envoyée dans le cloud. Tous les modèles tournent localement sur votre GPU Apple Silicon. Vos créations restent les vôtres.
"""),
        TextPreset(label: "Long EN", text: """
Fluxforge Studio turns your Mac into a complete AI creative studio. Generate high-quality images and videos from text, train your own custom models, and manage your creative library — all locally on your Apple Silicon, with no cloud or subscription required.

FORGE YOUR IDEA
A full creative workshop for exploring your visual ideas. Describe your concept in text, import an image or audio, then iterate freely: variations, style changes, video animations. Every step is saved in a Git-style branch tree — nothing is lost, everything is recoverable.

ADVANCED IMAGE GENERATION
Four Flux 2 models to choose from based on your needs. Adjust resolution, inference steps, guidance, use reference images, and enable automatic prompt enhancement.

VIDEO GENERATION
Create videos from text or images using the LTX-2.3 model. Two variants: Distilled (fast) and Dev (high quality). Add an automatically generated soundtrack.

100% LOCAL AND PRIVATE
No account required. No data sent to the cloud. All models run locally on your Apple Silicon GPU. Your creations remain yours.
"""),
    ]

    // MARK: - Model Loading

    func loadModel() async {
        let modelId = selectedModelId
        guard let modelInfo = VoxtralTTSRegistry.model(withId: modelId) else {
            log("Unknown model: \(modelId)")
            return
        }

        if isModelLoaded && currentModelId == modelId { return }

        isLoading = true
        isModelLoaded = false
        loadProgress = 0
        loadStatus = "Loading \(modelInfo.name)..."
        log("Loading \(modelInfo.name)...")

        pipeline?.unload()

        var config = VoxtralTTSPipeline.Configuration.default
        config.sanitizeText = sanitizeEnabled
        pipeline = VoxtralTTSPipeline(configuration: config)

        do {
            try await pipeline!.loadModel(modelInfo: modelInfo) { [weak self] progress, status in
                Task { @MainActor in
                    self?.loadProgress = progress
                    self?.loadStatus = status
                }
            }
            currentModelId = modelId
            currentModelName = modelInfo.name
            isModelLoaded = true
            log("Model loaded: \(modelInfo.name)")
        } catch {
            log("Load failed: \(error.localizedDescription)")
        }

        isLoading = false
    }

    // MARK: - Streaming Playback

    func startStreaming() {
        guard isModelLoaded, let pipeline, !isSynthesizing else { return }
        guard let voice = VoxtralVoice(rawValue: selectedVoice) else {
            log("Unknown voice: \(selectedVoice)")
            return
        }

        // Update sanitize setting
        pipeline.configuration.sanitizeText = sanitizeEnabled

        // Reset metrics
        ttft = nil
        totalTime = nil
        audioDuration = 0
        rtf = nil
        fps = 0
        framesGenerated = 0
        chunksReceived = 0
        isSynthesizing = true

        let clickTime = Date()
        buttonClickTime = clickTime
        log("--- Play clicked at \(formatTime(clickTime)) ---")
        log("Text: \"\(text.prefix(80))...\"")
        log("Model: \(currentModelName), Voice: \(selectedVoice), Sanitize: \(sanitizeEnabled ? "ON" : "OFF")")

        setupAudioEngine()

        streamingTask = Task {
            var totalSamplesScheduled = 0

            do {
                let stream = pipeline.synthesizeStreaming(
                    text: text,
                    voice: voice,
                    chunkSize: 10
                )

                for try await chunk in stream {
                    if Task.isCancelled { break }

                    if chunk.isFirst {
                        let ttftMs = chunk.elapsed * 1000
                        self.ttft = chunk.elapsed
                        self.log("TTFT: \(String(format: "%.0f", ttftMs)) ms (first \(chunk.frameCount) frames)")
                    }

                    scheduleAudioChunk(chunk.waveform)
                    totalSamplesScheduled += chunk.waveform.dim(0)

                    self.framesGenerated = chunk.totalFrames
                    self.chunksReceived += 1
                    self.audioDuration = Double(totalSamplesScheduled) / 24000.0
                    if chunk.elapsed > 0 {
                        self.fps = Double(chunk.totalFrames) / chunk.elapsed
                    }

                    if chunk.isFinal {
                        self.totalTime = chunk.elapsed
                        self.rtf = chunk.elapsed / self.audioDuration
                        self.log("Done: \(chunk.totalFrames) frames, \(String(format: "%.2f", self.audioDuration))s audio in \(String(format: "%.2f", chunk.elapsed))s")
                        self.log("RTF: \(String(format: "%.2f", self.rtf!))x, FPS: \(String(format: "%.1f", self.fps))")
                    }
                }
            } catch {
                self.log("Error: \(error.localizedDescription)")
            }

            self.isSynthesizing = false
        }
    }

    func stop() {
        streamingTask?.cancel()
        streamingTask = nil
        playerNode?.stop()
        audioEngine?.stop()
        isSynthesizing = false
        log("Stopped")
    }

    // MARK: - Audio Engine

    private func setupAudioEngine() {
        audioEngine?.stop()
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()

        let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!
        audioEngine!.attach(playerNode!)
        audioEngine!.connect(playerNode!, to: audioEngine!.mainMixerNode, format: format)

        do {
            try audioEngine!.start()
            playerNode!.play()
        } catch {
            log("Audio engine error: \(error.localizedDescription)")
        }
    }

    private func scheduleAudioChunk(_ waveform: MLXArray) {
        guard let playerNode else { return }

        let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!
        let samples = waveform.asType(.float32)
        MLX.eval(samples)
        let floatArray = samples.asArray(Float.self)

        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: UInt32(floatArray.count))!
        buffer.frameLength = UInt32(floatArray.count)
        floatArray.withUnsafeBufferPointer { ptr in
            buffer.floatChannelData![0].update(from: ptr.baseAddress!, count: floatArray.count)
        }
        playerNode.scheduleBuffer(buffer)
    }

    // MARK: - Logging

    static let logFileURL: URL = {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral_streaming_bench.log")
        try? "".write(to: url, atomically: true, encoding: .utf8)
        return url
    }()

    private func log(_ message: String) {
        let ts = formatTime(Date())
        let line = "[\(ts)] \(message)"
        logLines.append(line)
        if let data = (line + "\n").data(using: .utf8),
           let fh = try? FileHandle(forWritingTo: Self.logFileURL) {
            fh.seekToEndOfFile()
            fh.write(data)
            fh.closeFile()
        }
        print(line)
    }

    private func formatTime(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f.string(from: date)
    }
}
