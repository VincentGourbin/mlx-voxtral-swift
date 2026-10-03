/**
 * EvalCommand - `voxtral eval`: reproducible quality measurement (K-33).
 *
 * Transcribes the clips of a declared corpus (`docs/eval/corpus.json`: audio, language, exact reference text, both
 * SHA-256) and prints one `EVAL {json}` line per clip (also appended to `<out>/eval.jsonl`): normalized WER (see
 * `WER.swift`), word counts, length ratio, last reference sentence present, required sentences present, output
 * SHA-256. Same refusals as `bench` (Debug binary, busy machine) and a corpus whose files do not match their SHA-256
 * is refused. Greedy decoding: two runs on the same build give the same scores.
 */

import ArgumentParser
import Foundation
import VoxtralCore

struct Eval: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "eval",
        abstract: "Measure transcription quality on a declared corpus: one EVAL JSON line per clip (normalized WER)",
        subcommands: [EvalSTT.self, EvalRealtime.self]
    )
}

struct EvalCorpus: Decodable {
    struct Clip: Decodable {
        let id: String
        let audio: String
        /// ISO code, or "mixed" for a bilingual clip (always transcribed with auto-detection)
        let language: String
        let reference: String
        let audioSHA256: String
        let referenceSHA256: String
        /// Sentences that must appear in the transcription (casing, punctuation and accents ignored)
        let mustContain: [String]?

        enum CodingKeys: String, CodingKey {
            case id, audio, language, reference
            case audioSHA256 = "audio_sha256"
            case referenceSHA256 = "reference_sha256"
            case mustContain = "must_contain"
        }
    }

    let clips: [Clip]

    /// The selected clips, each with its reference text, after checking both SHA-256
    static func load(path: String, ids: String?) throws -> [(clip: Clip, reference: String)] {
        let corpus = try JSONDecoder().decode(EvalCorpus.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        let wanted = ids.map { Set($0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }) }
        let selected = corpus.clips.filter { wanted?.contains($0.id) ?? true }
        if let wanted, selected.count != wanted.count {
            throw ValidationError("Unknown clip id(s): \(wanted.subtracting(selected.map(\.id)).sorted())")
        }
        return try selected.map { clip in
            for (file, expected) in [(clip.audio, clip.audioSHA256), (clip.reference, clip.referenceSHA256)] {
                guard let data = FileManager.default.contents(atPath: file) else {
                    print("REFUSED corpus: \(file) not found (clip \(clip.id))")
                    throw ExitCode(4)
                }
                let actual = BenchJSON.sha256(data)
                guard actual == expected else {
                    print("REFUSED corpus: \(file) sha256 \(actual) ≠ \(expected) (clip \(clip.id))")
                    throw ExitCode(4)
                }
            }
            return (clip, try String(contentsOfFile: clip.reference, encoding: .utf8))
        }
    }
}

struct EvalCommonOptions: ParsableArguments {
    @Option(name: .long, help: "Corpus file") var corpus: String = "docs/eval/corpus.json"
    @Option(name: .long, help: "Comma-separated clip ids (default: every clip)") var clips: String?
    @Option(name: .long, help: "Output directory for eval.jsonl") var out: String = ".local-runs/eval.noindex"
    @Option(name: .long, help: "Run label written in each line (e.g. run1, run2)") var tag: String = "run1"
    @Flag(name: .long, help: "Advertise activity to external monitors (SiliconScope)") var beacon: Bool = false
}

enum EvalRunner {
    /// Scores one transcription and prints its EVAL line
    static func report(
        _ base: [String: Any], clip: EvalCorpus.Clip, reference: String, hypothesis: String,
        truncated: Bool, common: EvalCommonOptions
    ) {
        let score = WER.score(reference: reference, hypothesis: hypothesis)
        var record = base
        record["clip"] = clip.id
        record["input"] = clip.audio
        record["input_s"] = BenchJSON.round(BenchRunner.audioSeconds(clip.audio), 2) ?? 0
        record["clip_language"] = clip.language
        record["wer"] = BenchJSON.round(score.wer * 100, 2) ?? 0   // percent
        record["ref_words"] = score.referenceWords
        record["hyp_words"] = score.hypothesisWords
        record["sub"] = score.substitutions
        record["del"] = score.deletions
        record["ins"] = score.insertions
        record["length_ratio"] = BenchJSON.round(
            score.referenceWords == 0 ? 0 : Double(score.hypothesisWords) / Double(score.referenceWords), 3) ?? 0
        record["last_sentence_present"] = WER.contains(hypothesis, sentence: WER.lastSentence(of: reference))
        if let required = clip.mustContain {
            record["must_contain_present"] = required.allSatisfy { WER.contains(hypothesis, sentence: $0) }
        }
        record["truncated"] = truncated
        record["out_sha256"] = BenchJSON.sha256(Data(hypothesis.utf8))
        record["reference_sha256"] = clip.referenceSHA256
        record["tag"] = common.tag
        BenchJSON.emit(record, out: common.out, tag: "EVAL", file: "eval.jsonl")
    }
}

// MARK: - stt

struct EvalSTT: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "stt", abstract: "VoxtralPipeline.transcribe (greedy)")
    @Option(name: .long) var model: String = "mini-3b-8bit"
    @Option(name: .long, help: "mlx | auto") var backend: String = "mlx"
    @Option(name: .long, help: "explicit (the clip's language) | auto (language: nil)") var languageMode: String = "explicit"
    @OptionGroup var common: EvalCommonOptions

    func run() async throws {
        try BenchRunner.guardBuild()
        guard ["explicit", "auto"].contains(languageMode) else { throw ValidationError("--language-mode explicit|auto") }
        guard let pipelineModel = parseSTTModelID(model) else { throw ValidationError("Unknown STT model: \(model)") }
        let clips = try EvalCorpus.load(path: common.corpus, ids: common.clips)
        try BenchRunner.prepare(pass: 1, cooldown: 0)
        if common.beacon { VoxtralRuntimeBeacon.isEnabled = true }
        let pipeline = VoxtralPipeline(model: pipelineModel, backend: try parseBackend(backend))
        try await pipeline.loadModel()
        for (clip, reference) in clips {
            let language = languageMode == "explicit" && clip.language != "mixed" ? clip.language : nil
            let text = try await pipeline.transcribe(audio: URL(fileURLWithPath: clip.audio), language: language)
            let base: [String: Any] = [
                "pipeline": "stt", "model": model, "backend": backend, "language": language ?? "auto",
                "language_mode": languageMode,
            ]
            EvalRunner.report(base, clip: clip, reference: reference, hypothesis: text,
                              truncated: pipeline.lastResultTruncated, common: common)
        }
        pipeline.unload()
    }
}

// MARK: - realtime

struct EvalRealtime: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "realtime", abstract: "VoxtralRealtimePipeline.transcribe (greedy)")
    @Option(name: .long) var model: String = "realtime-4b-4bit"
    @Option(name: .long, help: "Transcription delay (ms)") var delay: Int = 480
    @OptionGroup var common: EvalCommonOptions

    func run() async throws {
        try BenchRunner.guardBuild()
        let clips = try EvalCorpus.load(path: common.corpus, ids: common.clips)
        try BenchRunner.prepare(pass: 1, cooldown: 0)
        if common.beacon { VoxtralRuntimeBeacon.isEnabled = true }
        let pipeline = VoxtralRealtimePipeline(configuration: .init(transcriptionDelayMs: delay))
        try await pipeline.loadModel(modelId: model)
        for (clip, reference) in clips {
            let text = try await pipeline.transcribe(audio: URL(fileURLWithPath: clip.audio))
            let base: [String: Any] = ["pipeline": "realtime", "model": model, "delay_ms": delay, "language": "auto"]
            EvalRunner.report(base, clip: clip, reference: reference, hypothesis: text,
                              truncated: pipeline.lastTranscriptionTruncated, common: common)
        }
        pipeline.unload()
    }
}
