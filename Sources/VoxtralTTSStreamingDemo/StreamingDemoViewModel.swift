import Foundation
import AVFoundation
import AppKit
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

    /// Auto-save each synthesized audio to disk (for regression A/B vs baselines).
    @Published var saveCaptures: Bool = true
    /// Path of the most recently saved capture (for "reveal in Finder").
    @Published var lastCaptureURL: URL?

    /// Seed for reproducible synthesis. A fixed default makes runs deterministic
    /// (same text+voice → same audio); clear the field for the old random draw.
    @Published var seedText: String = "42"

    // MARK: - Voice cloning inputs

    @Published var referenceURL: URL?
    @Published var cloneName: String = "my_voice"
    @Published var cloneDuration: Double = 16
    @Published var cloneEpochs: Int = 3000
    @Published var isEnrolling = false
    @Published var enrollStatus: String = ""
    @Published var enrollProgress: Double = 0        // 0…1 over epochs
    @Published var clonedVoices: [ClonedVoice] = []

    struct ClonedVoice: Identifiable, Hashable {
        let name: String
        let url: URL
        var id: String { "cloned:\(name)" }
    }

    // MARK: - Reference builder (video/audio → extracts → reference)

    struct RefExtract: Identifiable, Hashable {
        let id = UUID()
        let start: Double
        let end: Double
        var duration: Double { end - start }
    }

    @Published var refSourceURL: URL?
    @Published var refSourceDuration: Double = 0
    @Published var refExtracts: [RefExtract] = []
    @Published var segStart: Double = 0
    @Published var segEnd: Double = 8
    @Published var refBuilderBusy = false
    @Published var refBuilderStatus: String = ""
    let ffmpegAvailable = FFmpeg.isAvailable

    var refExtractsTotal: Double { refExtracts.reduce(0) { $0 + $1.duration } }
    private var previewPlayer: AVAudioPlayer?
    private static let refWorkDir: URL = {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("VoxtralRefBuilder", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }()

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
        ("tts-4b", "bf16 original (8 GB)"),
        ("tts-4b-mlx", "bf16 MLX (8 GB)")
    ]

    let presetVoices: [(id: String, label: String)] = VoxtralVoice.allCases.map {
        ($0.rawValue, $0.displayName)
    }

    /// Preset voices plus any enrolled (cloned) voices, for the voice picker.
    var voicePickerOptions: [(id: String, label: String)] {
        presetVoices + clonedVoices.map { ($0.id, "🎙️ \($0.name) (cloned)") }
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

    // MARK: - Mic recording (read a prompt → reference)

    struct RecordPrompt: Identifiable, Hashable {
        let id = UUID()
        let lang: String
        let text: String
    }

    /// Prompts sized to read in roughly the target reference length (~16 s).
    let recordPrompts: [RecordPrompt] = [
        RecordPrompt(lang: "EN", text: "The rapid development of artificial intelligence is reshaping how we live and work. From the way we search for information to how we create images and music, these tools are quietly becoming part of our everyday lives."),
        RecordPrompt(lang: "FR", text: "Le développement rapide de l'intelligence artificielle transforme notre façon de vivre et de travailler. De la manière dont nous cherchons l'information à celle dont nous créons des images et de la musique, ces outils s'installent peu à peu dans notre quotidien."),
        RecordPrompt(lang: "EN", text: "Good morning. Today I want to talk about something simple but important: the value of taking your time. In a world that rewards speed, slowing down to think clearly is a quiet kind of strength that pays off in the long run."),
    ]

    @Published var recordPromptIndex = 0
    @Published var isRecording = false
    @Published var recordElapsed: Double = 0
    @Published var micStatus: String = ""

    private var recorder: AVAudioRecorder?
    private var recordTimer: Timer?
    private var recordStart: Date?

    func startRecording() {
        guard !isRecording else { return }
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
            Task { @MainActor in
                guard let self else { return }
                guard granted else { self.micStatus = "Microphone access denied"; return }
                self.beginRecording()
            }
        }
    }

    private func beginRecording() {
        let url = Self.refWorkDir.appendingPathComponent("mic_\(UUID().uuidString).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 24000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
        ]
        do {
            let rec = try AVAudioRecorder(url: url, settings: settings)
            rec.record()
            recorder = rec
            recordStart = Date()
            recordElapsed = 0
            isRecording = true
            micStatus = "Recording…"
            recordTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    guard let self, let s = self.recordStart else { return }
                    self.recordElapsed = Date().timeIntervalSince(s)
                }
            }
        } catch {
            micStatus = "Recorder error: \(error.localizedDescription)"
        }
    }

    /// Stop recording. If long enough, set it as the enrollment reference.
    func stopRecording() {
        guard isRecording else { return }
        recorder?.stop()
        recordTimer?.invalidate(); recordTimer = nil
        isRecording = false
        let url = recorder?.url
        let elapsed = recordElapsed
        recorder = nil
        if let url, elapsed >= cloneDuration {
            referenceURL = url
            micStatus = String(format: "Reference recorded (%.1f s)", elapsed)
            log("Recorded reference from mic (\(String(format: "%.1f", elapsed))s)")
        } else {
            micStatus = String(format: "Too short (%.1f s, need %.0f s)", elapsed, cloneDuration)
        }
    }

    // MARK: - Reference builder actions

    /// Load a video/audio file and read its total duration (via ffprobe).
    func loadRefSource(_ url: URL) {
        refSourceURL = url
        refExtracts = []
        refSourceDuration = 0
        segStart = 0
        segEnd = min(cloneDuration, 8)
        refBuilderStatus = "Reading \(url.lastPathComponent)…"
        Task {
            do {
                let dur = try await FFmpeg.duration(of: url)
                await MainActor.run {
                    self.refSourceDuration = dur
                    self.segEnd = min(self.segStart + self.cloneDuration, dur)
                    self.refBuilderStatus = String(format: "Source: %.0f s", dur)
                }
            } catch {
                await MainActor.run { self.refBuilderStatus = "Error: \(error.localizedDescription)" }
            }
        }
    }

    /// Preview the current [segStart, segEnd] selection (extract + play).
    func previewSegment() {
        guard let src = refSourceURL, segEnd > segStart else { return }
        let (start, end) = (segStart, segEnd)
        refBuilderStatus = "Extracting preview…"
        Task {
            do {
                let out = Self.refWorkDir.appendingPathComponent("preview.wav")
                try await FFmpeg.extractSegment(from: src, start: start, end: end, to: out)
                let data = try Data(contentsOf: out)
                await MainActor.run {
                    self.previewPlayer = try? AVAudioPlayer(data: data)
                    self.previewPlayer?.play()
                    self.refBuilderStatus = String(format: "Preview %.1f–%.1f s", start, end)
                }
            } catch {
                await MainActor.run { self.refBuilderStatus = "Error: \(error.localizedDescription)" }
            }
        }
    }

    func addExtract() {
        guard segEnd > segStart else { return }
        refExtracts.append(RefExtract(start: segStart, end: segEnd))
        // Advance the selector past this extract for convenience.
        let next = min(segEnd, refSourceDuration)
        segStart = next
        segEnd = min(next + cloneDuration, refSourceDuration)
    }

    func removeExtract(_ id: RefExtract.ID) {
        refExtracts.removeAll { $0.id == id }
    }

    /// Concatenate the chosen extracts into a single reference WAV and set it
    /// as the enrollment reference. Returns via `referenceURL`.
    func buildReference() {
        guard let src = refSourceURL, !refExtracts.isEmpty, !refBuilderBusy else { return }
        refBuilderBusy = true
        refBuilderStatus = "Building reference…"
        let extracts = refExtracts
        Task {
            do {
                var parts: [URL] = []
                for (i, e) in extracts.enumerated() {
                    let part = Self.refWorkDir.appendingPathComponent("part_\(i).wav")
                    try await FFmpeg.extractSegment(from: src, start: e.start, end: e.end, to: part)
                    parts.append(part)
                }
                let out = Self.refWorkDir.appendingPathComponent("reference_\(UUID().uuidString).wav")
                try await FFmpeg.concat(parts, to: out)
                await MainActor.run {
                    self.referenceURL = out
                    self.refBuilderBusy = false
                    self.refBuilderStatus = String(format: "Reference ready (%.1f s)", self.refExtractsTotal)
                    self.log("Built reference from \(extracts.count) extract(s), \(String(format: "%.1f", self.refExtractsTotal))s")
                }
            } catch {
                await MainActor.run {
                    self.refBuilderBusy = false
                    self.refBuilderStatus = "Error: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Voice Cloning

    /// Directory where enrolled (cloned) voice embeddings are stored.
    static let clonedVoicesDir: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = base.appendingPathComponent("VoxtralClonedVoices", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    /// Load the list of previously enrolled voices from disk.
    func refreshClonedVoices() {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: Self.clonedVoicesDir, includingPropertiesForKeys: nil)) ?? []
        clonedVoices = files
            .filter { $0.pathExtension == "safetensors" }
            .map { ClonedVoice(name: $0.deletingPathExtension().lastPathComponent, url: $0) }
            .sorted { $0.name < $1.name }
    }

    /// Enroll a voice from `referenceURL` using the currently loaded model.
    /// Runs the (long) optimization off the main actor and streams progress.
    func enroll() {
        guard isModelLoaded, let pipeline, !isEnrolling, !isSynthesizing else { return }
        guard let ref = referenceURL else { log("No reference audio selected"); return }
        let name = cloneName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { log("Enter a name for the cloned voice"); return }

        isEnrolling = true
        enrollProgress = 0
        enrollStatus = "Preparing…"
        let epochs = cloneEpochs
        let duration = cloneDuration
        let outURL = Self.clonedVoicesDir.appendingPathComponent("\(name).safetensors")
        log("--- Enrolling '\(name)' from \(ref.lastPathComponent) (\(epochs) epochs, \(Int(duration))s) ---")

        // The pipeline is not Sendable; box it to run the sync, GPU-heavy
        // enrollment off the main actor without blocking the UI.
        final class Box: @unchecked Sendable { let p: VoxtralTTSPipeline; init(_ p: VoxtralTTSPipeline) { self.p = p } }
        let box = Box(pipeline)

        Task.detached { [weak self] in
            var config = VoxtralVoiceEnrollment.Config()
            config.numFrames = Int(duration * 12.5)
            config.epochs = epochs
            config.logEvery = 100
            do {
                try box.p.enrollVoice(referenceURL: ref, outputURL: outURL, config: config) { progress in
                    // Extract Sendable value types before hopping to the main actor.
                    let epoch = progress.epoch
                    let loss = progress.totalLoss
                    Task { @MainActor in
                        self?.enrollProgress = Double(epoch) / Double(epochs)
                        self?.enrollStatus = "epoch \(epoch)/\(epochs) · loss \(String(format: "%.3f", loss))"
                        // Also record to the log file so the loss curve is
                        // observable outside the UI.
                        self?.log("enroll epoch \(epoch)/\(epochs) loss \(String(format: "%.4f", loss))")
                    }
                }
                await MainActor.run {
                    self?.log("Voice enrolled: \(name)")
                    self?.enrollStatus = "Done"
                    self?.isEnrolling = false
                    self?.refreshClonedVoices()
                    self?.selectedVoice = "cloned:\(name)"
                }
            } catch {
                await MainActor.run {
                    self?.log("Enrollment failed: \(error.localizedDescription)")
                    self?.enrollStatus = "Failed"
                    self?.isEnrolling = false
                }
            }
        }
    }

    private func loadClonedEmbedding(_ url: URL) throws -> MLXArray {
        let arrays = try MLX.loadArrays(url: url)
        guard let embedding = arrays["embedding"] else {
            throw VoxtralTTSError.invalidConfiguration("No 'embedding' array in \(url.lastPathComponent)")
        }
        return embedding
    }

    // MARK: - Streaming Playback

    func startStreaming() {
        guard isModelLoaded, let pipeline, !isSynthesizing else { return }

        // Resolve the selected voice: a cloned voice (embedding) or a preset.
        let clonedVoice = clonedVoices.first { $0.id == selectedVoice }
        var voiceEmbedding: MLXArray?
        var presetVoice: VoxtralVoice?
        if let clonedVoice {
            do { voiceEmbedding = try loadClonedEmbedding(clonedVoice.url) }
            catch { log("Failed to load cloned voice: \(error.localizedDescription)"); return }
        } else if let v = VoxtralVoice(rawValue: selectedVoice) {
            presetVoice = v
        } else {
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
        capturedSamples.removeAll(keepingCapacity: true)

        let clickTime = Date()
        buttonClickTime = clickTime
        log("--- Play clicked at \(formatTime(clickTime)) ---")
        log("Text: \"\(text.prefix(80))...\"")
        log("Model: \(currentModelName), Voice: \(selectedVoice), Sanitize: \(sanitizeEnabled ? "ON" : "OFF")")

        setupAudioEngine()

        streamingTask = Task {
            var totalSamplesScheduled = 0

            do {
                // Reproducible seed (empty field → random, the old behavior).
                let seed = UInt64(self.seedText.trimmingCharacters(in: .whitespaces))
                // Warm-up vocalise stabilizes enrolled (cloned) voices (A6b);
                // presets don't need it.
                let warmUp = (clonedVoice != nil) ? VoxtralTTSPipeline.recommendedWarmUpVocalise : nil

                let stream: AsyncThrowingStream<TTSStreamingChunk, Error>
                if let voiceEmbedding {
                    stream = pipeline.synthesizeStreaming(text: text, voiceEmbedding: voiceEmbedding, chunkSize: 10,
                                                          seed: seed, warmUpText: warmUp, warmUpLeadInFrames: 0)
                } else {
                    stream = pipeline.synthesizeStreaming(text: text, voice: presetVoice!, chunkSize: 10,
                                                          seed: seed, warmUpText: warmUp, warmUpLeadInFrames: 0)
                }

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

            // Persist the full synthesized waveform so a run can be A/B'd
            // against the saved baselines in docs/examples/.
            if self.saveCaptures { self.writeCaptureWAV(self.capturedSamples) }

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

        // Keep a copy for the on-disk capture (played chunks are otherwise lost).
        if saveCaptures { capturedSamples.append(contentsOf: floatArray) }

        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: UInt32(floatArray.count))!
        buffer.frameLength = UInt32(floatArray.count)
        floatArray.withUnsafeBufferPointer { ptr in
            buffer.floatChannelData![0].update(from: ptr.baseAddress!, count: floatArray.count)
        }
        playerNode.scheduleBuffer(buffer)
    }

    // MARK: - Capture (save synthesized audio to disk)

    /// Accumulated float32 samples of the current synthesis (24 kHz mono).
    private var capturedSamples: [Float] = []

    /// Directory where captured syntheses are written (visible in Finder).
    static let capturesDir: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = base.appendingPathComponent("VoxtralCaptures", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    /// Write `samples` (24 kHz mono) to a timestamped 16-bit PCM WAV.
    private func writeCaptureWAV(_ samples: [Float]) {
        guard !samples.isEmpty else { log("Capture skipped: no audio"); return }

        let ts = fileTimestamp()
        let model = (currentModelId ?? "model")
        let voice = selectedVoice.replacingOccurrences(of: ":", with: "-")
        let url = Self.capturesDir.appendingPathComponent("cap_\(model)_\(voice)_\(ts).wav")

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 24000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        do {
            let file = try AVAudioFile(forWriting: url, settings: settings)
            // Source buffer is float32; AVAudioFile converts to the 16-bit file format.
            let srcFormat = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!
            let buffer = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: UInt32(samples.count))!
            buffer.frameLength = UInt32(samples.count)
            samples.withUnsafeBufferPointer { ptr in
                buffer.floatChannelData![0].update(from: ptr.baseAddress!, count: samples.count)
            }
            try file.write(from: buffer)
            lastCaptureURL = url
            let secs = Double(samples.count) / 24000.0
            log("Saved capture (\(String(format: "%.2f", secs))s): \(url.path)")
        } catch {
            log("Capture save failed: \(error.localizedDescription)")
        }
    }

    /// Reveal the captures directory (or the last capture) in Finder.
    func revealCaptures() {
        NSWorkspace.shared.activateFileViewerSelecting(
            [lastCaptureURL ?? Self.capturesDir])
    }

    private func fileTimestamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmmss"
        return f.string(from: Date())
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
