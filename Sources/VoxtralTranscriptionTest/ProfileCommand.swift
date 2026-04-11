// ProfileCommand.swift - Profiling commands for Voxtral CLI
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import VoxtralCore
import MLX
import MLXProfiler

// MARK: - Profile Command Group

struct Profile: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "profile",
        abstract: "Profile Voxtral inference pipelines (STT, TTS, Realtime)",
        subcommands: [ProfileRun.self],
        defaultSubcommand: ProfileRun.self
    )
}

// MARK: - Pipeline Selection

enum PipelineType: String, ExpressibleByArgument, CaseIterable {
    case stt
    case chat
    case tts
    case realtime
}

// MARK: - Profile Run

@available(macOS 14.0, *)
struct ProfileRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Profile a single inference run with Chrome Trace export"
    )

    @Option(name: .long, help: "Pipeline to profile: stt, chat, tts, realtime")
    var pipeline: PipelineType = .stt

    @Option(name: .long, help: "Audio file path (required for stt/chat/realtime)")
    var audio: String?

    @Option(name: .long, help: "Text to synthesize (required for tts) or chat prompt (for chat)")
    var text: String?

    @Option(name: .long, help: "STT model: mini-3b, mini-3b-8bit, mini-3b-4bit, small-24b, small-24b-8bit, small-4bit")
    var model: String = "mini-3b-8bit"

    @Option(name: .long, help: "TTS model: tts-4b-mlx, tts-4b-4bit, tts-4b-6bit")
    var ttsModel: String = "tts-4b-mlx"

    @Option(name: .long, help: "Realtime model: realtime-4b-4bit, realtime-4b-fp16")
    var realtimeModel: String = "realtime-4b-4bit"

    @Option(name: .long, help: "Voice preset for TTS")
    var voice: String = "neutral_female"

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 500

    @Option(name: [.customShort("t"), .long], help: "Temperature (0.0 = greedy, default 0.7 for chat)")
    var temperature: Float?

    @Flag(name: .long, help: "Track memory per token/frame")
    var perStepMemory: Bool = false

    @Flag(name: .long, help: "Disable Chrome Trace export")
    var noChromeTrace: Bool = false

    @Option(name: .long, help: "Output directory for trace files")
    var output: String?

    func run() async throws {
        // Validate inputs
        switch pipeline {
        case .stt, .realtime:
            guard let audio, FileManager.default.fileExists(atPath: audio) else {
                throw ValidationError("--audio <path> required for \(pipeline.rawValue) pipeline")
            }
        case .chat:
            guard let audio, FileManager.default.fileExists(atPath: audio) else {
                throw ValidationError("--audio <path> required for chat pipeline")
            }
            guard let text, !text.isEmpty else {
                throw ValidationError("--text <prompt> required for chat pipeline")
            }
        case .tts:
            guard let text, !text.isEmpty else {
                throw ValidationError("--text <string> required for tts pipeline")
            }
        }

        // Create profiling session
        let config = ProfilingConfig(
            trackMemory: true,
            trackPerStepMemory: perStepMemory,
            exportChromeTrace: !noChromeTrace,
            printSummary: true
        )
        let session = ProfilingSession(config: config)
        session.title = "VOXTRAL \(pipeline.rawValue.uppercased()) PROFILING REPORT"
        session.metadata["pipeline"] = pipeline.rawValue

        let profiler = MLXProfiler.shared
        profiler.enable()
        profiler.activeSession = session
        defer {
            profiler.activeSession = nil
            profiler.disable()
        }

        print("Profiling Voxtral \(pipeline.rawValue.uppercased()) pipeline...")

        switch pipeline {
        case .stt:
            try await profileSTT(session: session)
        case .chat:
            try await profileChat(session: session)
        case .tts:
            try await profileTTS(session: session)
        case .realtime:
            try await profileRealtime(session: session)
        }

        // Print report
        if config.printSummary {
            print(session.generateReport())
        }

        // Print LLM/TTS metrics
        switch pipeline {
        case .stt, .chat, .realtime:
            let llmMetrics = profiler.getLLMMetrics()
            print("\nLLM Metrics:")
            print(llmMetrics.summary)
        case .tts:
            let ttsMetrics = profiler.getTTSMetrics()
            print("\nTTS Metrics:")
            print(ttsMetrics.summary)
        }

        // Export Chrome Trace
        if !noChromeTrace {
            let traceData = ChromeTraceExporter.export(session: session)
            let outputDir = output.map { URL(fileURLWithPath: $0) }
                ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            let fileName = "voxtral_\(pipeline.rawValue)_trace.json"
            let traceURL = outputDir.appendingPathComponent(fileName)
            try traceData.write(to: traceURL)
            print("\nChrome Trace: \(traceURL.path)")
            print("Open in https://ui.perfetto.dev/")
        }
    }

    // MARK: - STT Profiling

    private func profileSTT(session: ProfilingSession) async throws {
        guard let audio else { return }

        guard let pipelineModel = parseSTTModel(model) else {
            throw ValidationError("Unknown STT model: \(model)")
        }

        session.metadata["model"] = model
        print("Model: \(pipelineModel.displayName)")

        var config = VoxtralPipeline.Configuration.default
        config.maxTokens = maxTokens
        config.temperature = temperature ?? 0.0

        let sttPipeline = VoxtralPipeline(model: pipelineModel, configuration: config)

        try await sttPipeline.loadModel { progress, status in
            print("  [\(Int(progress * 100))%] \(status)")
        }

        let audioURL = URL(fileURLWithPath: audio)
        let transcription = try await sttPipeline.transcribe(audio: audioURL)

        print("\nResult: \(transcription.prefix(200))...")
        session.metadata["resultLength"] = "\(transcription.count)"

        sttPipeline.unload()
    }

    // MARK: - Chat Profiling

    private func profileChat(session: ProfilingSession) async throws {
        guard let audio, let text else { return }

        guard let pipelineModel = parseSTTModel(model) else {
            throw ValidationError("Unknown STT model: \(model)")
        }

        session.metadata["model"] = model
        session.metadata["prompt"] = String(text.prefix(100))
        print("Model: \(pipelineModel.displayName)")
        print("Prompt: \(text)")

        var config = VoxtralPipeline.Configuration.default
        config.maxTokens = maxTokens
        config.temperature = temperature ?? 0.7

        let chatPipeline = VoxtralPipeline(model: pipelineModel, configuration: config)

        try await chatPipeline.loadModel { progress, status in
            print("  [\(Int(progress * 100))%] \(status)")
        }

        let audioURL = URL(fileURLWithPath: audio)
        let response = try await chatPipeline.chat(audio: audioURL, prompt: text)

        print("\nResult: \(response.prefix(200))...")
        session.metadata["resultLength"] = "\(response.count)"

        chatPipeline.unload()
    }

    // MARK: - TTS Profiling

    private func profileTTS(session: ProfilingSession) async throws {
        guard let text else { return }

        guard let ttsModelInfo = VoxtralTTSRegistry.model(withId: ttsModel) else {
            throw ValidationError("Unknown TTS model: \(ttsModel)")
        }

        session.metadata["model"] = ttsModel
        session.metadata["voice"] = voice
        session.metadata["text_length"] = "\(text.count)"
        print("Model: \(ttsModelInfo.name)")
        print("Voice: \(voice)")

        var config = VoxtralTTSPipeline.Configuration.default
        config.maxFrames = maxTokens

        let ttsPipeline = VoxtralTTSPipeline(configuration: config)

        try await ttsPipeline.loadModel(modelInfo: ttsModelInfo) { progress, status in
            print("  [\(Int(progress * 100))%] \(status)")
        }

        guard let voicePreset = VoxtralVoice(rawValue: voice) else {
            throw ValidationError("Unknown voice: \(voice)")
        }

        let result = try await ttsPipeline.synthesize(text: text, voice: voicePreset)

        let profiler = MLXProfiler.shared
        let audioDuration = Double(result.waveform.dim(0)) / Double(ttsPipeline.sampleRate)
        profiler.setAudioDuration(audioDuration)

        session.metadata["numFrames"] = "\(result.numFrames)"
        session.metadata["audioDuration"] = String(format: "%.2f", audioDuration)
        session.metadata["ttft"] = String(format: "%.0f", result.timeToFirstToken * 1000)

        print("\nResult: \(result.numFrames) frames, \(String(format: "%.2f", audioDuration))s audio")
        print("TTFT: \(String(format: "%.0f", result.timeToFirstToken * 1000))ms")
        print("RT factor: \(String(format: "%.2f", audioDuration / result.generationTime))x")

        ttsPipeline.unload()
    }

    // MARK: - Realtime Profiling

    private func profileRealtime(session: ProfilingSession) async throws {
        guard let audio else { return }

        session.metadata["model"] = realtimeModel
        print("Model: \(realtimeModel)")

        var config = VoxtralRealtimePipeline.Configuration.default
        config.maxTokens = maxTokens

        let realtimePipeline = VoxtralRealtimePipeline(configuration: config)

        try await realtimePipeline.loadModel(modelId: realtimeModel) { progress, status in
            print("  [\(Int(progress * 100))%] \(status)")
        }

        let audioURL = URL(fileURLWithPath: audio)
        let transcription = try await realtimePipeline.transcribe(audio: audioURL)

        print("\nResult: \(transcription.prefix(200))...")
        session.metadata["resultLength"] = "\(transcription.count)"

        realtimePipeline.unload()
    }

    // MARK: - Model Parsing

    private func parseSTTModel(_ id: String) -> VoxtralPipeline.Model? {
        switch id.lowercased() {
        case "mini-3b", "mini3b": return .mini3b
        case "mini-3b-8bit", "mini3b8bit": return .mini3b8bit
        case "mini-3b-4bit", "mini3b4bit": return .mini3b4bit
        case "small-24b", "small24b": return .small24b
        case "small-24b-8bit", "small24b8bit": return .small24b8bit
        case "small-4bit", "small4bit": return .small4bit
        default: return nil
        }
    }
}
