import Foundation
import AVFoundation
import VoxtralCore
import MLX

@available(macOS 14.0, *)
@MainActor
final class StreamingDemoViewModel: ObservableObject {

    // MARK: - User inputs

    @Published var text: String = "Voice was humanity's first interface. Long before writing or typing, it let us share ideas, coordinate work, and build relationships."
    @Published var selectedModelId: String = "tts-4b-4bit"
    @Published var selectedVoice: String = "neutral_male"

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

    // MARK: - Model Loading

    func loadModel() async {
        let modelId = selectedModelId
        guard let modelInfo = VoxtralTTSRegistry.model(withId: modelId) else {
            log("Unknown model: \(modelId)")
            return
        }

        // If same model already loaded, skip
        if isModelLoaded && currentModelId == modelId { return }

        isLoading = true
        isModelLoaded = false
        loadProgress = 0
        loadStatus = "Loading \(modelInfo.name)..."
        log("Loading \(modelInfo.name)...")

        // Unload previous
        pipeline?.unload()
        pipeline = VoxtralTTSPipeline()

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

        // Reset metrics
        ttft = nil
        totalTime = nil
        audioDuration = 0
        rtf = nil
        framesGenerated = 0
        chunksReceived = 0
        isSynthesizing = true

        // Record button click
        let clickTime = Date()
        buttonClickTime = clickTime
        log("--- Play clicked at \(formatTime(clickTime)) ---")
        log("Text: \"\(text.prefix(60))...\"")
        log("Model: \(currentModelName), Voice: \(selectedVoice)")

        // Setup audio engine
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

                    // Schedule audio on player
                    scheduleAudioChunk(chunk.waveform)
                    totalSamplesScheduled += chunk.waveform.dim(0)

                    // Update metrics
                    self.framesGenerated = chunk.totalFrames
                    self.chunksReceived += 1
                    self.audioDuration = Double(totalSamplesScheduled) / 24000.0

                    if chunk.isFinal {
                        self.totalTime = chunk.elapsed
                        self.rtf = chunk.elapsed / self.audioDuration
                        self.log("Done: \(chunk.totalFrames) frames, \(String(format: "%.2f", self.audioDuration))s audio in \(String(format: "%.2f", chunk.elapsed))s (RTF \(String(format: "%.2f", self.rtf!))x)")
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
        // Clear previous log
        try? "".write(to: url, atomically: true, encoding: .utf8)
        return url
    }()

    private func log(_ message: String) {
        let ts = formatTime(Date())
        let line = "[\(ts)] \(message)"
        logLines.append(line)
        // Also write to file for retrieval
        if let data = (line + "\n").data(using: .utf8),
           let fh = try? FileHandle(forWritingTo: Self.logFileURL) {
            fh.seekToEndOfFile()
            fh.write(data)
            fh.closeFile()
        }
        // Also print to stdout
        print(line)
    }

    private func formatTime(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f.string(from: date)
    }
}
