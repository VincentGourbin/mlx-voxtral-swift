/**
 * TTSGenerateStabilityReproTests — investigation harness for Fluxforge ask
 * A6a/A6b/A6c (custom-voice `generate` path, no prefix cache).
 *
 * Three experimental groups over N consecutive syntheses of the SAME text +
 * embedding, to separate the reported phenomena:
 *
 *   SEED  — MLXRandom.seed(k) before each run, no clearCache.
 *   NONE  — no seed, no clearCache (the app's default path).
 *   CLEAR — no seed, Memory.clearCache() before each run.
 *
 * What each group discriminates:
 *  - If SEED runs are bit-identical while the GPU cache grows between them,
 *    then output does NOT depend on cache-pool contents → A6a's
 *    "uninitialised buffer served from cache" mechanism is refuted.
 *  - If NONE runs vary (EOA frame + peak dBFS) with a fixed seed removed,
 *    the variance source is the RNG (flow-matching noise) → A6c.
 *  - If CLEAR runs vary just as much as NONE, clearCache does NOT fix the
 *    variance — it only bounds cache growth → confirms the A6a "fix" is
 *    incidental, not causal.
 *
 * WAVs are written for every run so the first-sentence warm-up (A6b) can be
 * listened to / analysed offline.
 *
 * Heavy (real 4B model). Skipped unless VOXTRAL_TTS_REPRO=1, with:
 *   VOXTRAL_TTS_REPRO_EMB    enrolled-voice .safetensors (required)
 *   VOXTRAL_TTS_REPRO_MODEL  model dir (optional; else registry default)
 *   VOXTRAL_TTS_REPRO_OUT    WAV output dir (optional; else NSTemporaryDirectory)
 *   VOXTRAL_TTS_REPRO_RUNS   runs per group (optional; default 5)
 */

import XCTest
import MLX
@testable import VoxtralCore

final class TTSGenerateStabilityReproTests: XCTestCase {

    private struct RunStats {
        let run: Int
        let numFrames: Int
        let samples: Int
        let maxAbs: Float
        let rms: Float
        let waveSum: Float
        let cacheBefore: Int
        let cacheAfter: Int
        let activeAfter: Int
        var maxDb: Float { 20 * log10(max(maxAbs, 1e-9)) }
        var rmsDb: Float { 20 * log10(max(rms, 1e-9)) }
    }

    private func mb(_ bytes: Int) -> Int { bytes / (1024 * 1024) }

    private func runOne(_ model: VoxtralTTSModel, _ tokenizer: TekkenTokenizer,
                        text: String, run: Int, seed: UInt64?, clear: Bool,
                        outDir: URL, tag: String) throws -> RunStats {
        if clear { Memory.clearCache() }
        let cacheBefore = Memory.cacheMemory
        // Exercise the real fix path: the seed: parameter on generate(), not
        // a manual MLXRandom.seed() at the call site.
        let (codes, numFrames, _) = model.generate(
            text: text, voiceEmbedding: currentEmb, tokenizer: tokenizer,
            maxTokens: 1500, sanitize: true, seed: seed)
        let waveform = model.decodeToWaveform(codes)
        MLX.eval(waveform)
        let f32 = waveform.asType(.float32)
        let maxAbs = MLX.abs(f32).max().item(Float.self)
        let rms = MLX.sqrt(MLX.mean(f32 * f32)).item(Float.self)
        let waveSum = f32.sum().item(Float.self)
        try WAVWriter.write(waveform: waveform,
                            to: outDir.appendingPathComponent("\(tag)_run\(run).wav"))
        let s = RunStats(run: run, numFrames: numFrames, samples: waveform.dim(0),
                         maxAbs: maxAbs, rms: rms, waveSum: waveSum,
                         cacheBefore: cacheBefore, cacheAfter: Memory.cacheMemory,
                         activeAfter: Memory.activeMemory)
        print(String(format: "[%@] run %d: EOA=%d samples=%d peak=%.4f (%.1f dBFS) rms=%.5f (%.1f dBFS) waveSum=%.5f | cache %d→%d MB active=%d MB",
                     tag, run, s.numFrames, s.samples, s.maxAbs, s.maxDb, s.rms, s.rmsDb, s.waveSum,
                     mb(s.cacheBefore), mb(s.cacheAfter), mb(s.activeAfter)))
        return s
    }

    private var currentEmb: MLXArray = MLXArray([Float(0)])

    func testGenerateStabilityAcrossConsecutiveRuns() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["VOXTRAL_TTS_REPRO"] == "1",
            "Set VOXTRAL_TTS_REPRO=1 to run this heavy repro test")

        let env = ProcessInfo.processInfo.environment
        let modelDir: URL
        if let override = env["VOXTRAL_TTS_REPRO_MODEL"] {
            modelDir = URL(fileURLWithPath: override)
        } else if let found = ModelDownloader.findTTSModelPath(for: VoxtralTTSRegistry.defaultModel) {
            modelDir = found
        } else {
            throw XCTSkip("Default TTS model not downloaded")
        }
        guard let embPath = env["VOXTRAL_TTS_REPRO_EMB"] else {
            throw XCTSkip("Set VOXTRAL_TTS_REPRO_EMB to an enrolled-voice .safetensors path")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: embPath))
        guard let voiceEmb = arrays["embedding"] ?? arrays.values.first else {
            throw XCTSkip("No embedding in \(embPath)")
        }
        currentEmb = voiceEmb
        let runs = env["VOXTRAL_TTS_REPRO_RUNS"].flatMap { Int($0) } ?? 5
        let outDir = URL(fileURLWithPath: env["VOXTRAL_TTS_REPRO_OUT"] ?? NSTemporaryDirectory())
            .appendingPathComponent("tts_a6")
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        // The app's exact A6 test text (3 French sentences).
        let text = """
        Du nouveau dans FluxForge Studio ! \
        Vous pouvez maintenant entraîner vos propres voix pour réaliser des doublages dignes d'une qualité professionnelle. \
        La version 3.1 apporte aussi son lot d'améliorations de performances.
        """

        print("[repro] embedding \(voiceEmb.shape), runs/group=\(runs), out=\(outDir.path)")
        let model = try loadVoxtralTTSModel(from: modelDir)
        let tokenizer = TekkenTokenizer(modelPath: modelDir.path)

        // Group SEED: fixed seed each run, no clearCache.
        var seedRuns: [RunStats] = []
        for r in 1...runs {
            seedRuns.append(try runOne(model, tokenizer, text: text, run: r,
                                       seed: 1234, clear: false, outDir: outDir, tag: "SEED"))
        }
        // Group NONE: no seed, no clearCache (app default).
        var noneRuns: [RunStats] = []
        for r in 1...runs {
            noneRuns.append(try runOne(model, tokenizer, text: text, run: r,
                                       seed: nil, clear: false, outDir: outDir, tag: "NONE"))
        }
        // Group CLEAR: no seed, clearCache before each.
        var clearRuns: [RunStats] = []
        for r in 1...runs {
            clearRuns.append(try runOne(model, tokenizer, text: text, run: r,
                                        seed: nil, clear: true, outDir: outDir, tag: "CLEAR"))
        }

        func spread(_ g: [RunStats]) -> (frames: String, peakDb: String) {
            let f = g.map { $0.numFrames }
            let p = g.map { $0.maxDb }
            return ("[\(f.map(String.init).joined(separator: ","))]",
                    p.map { String(format: "%.1f", $0) }.joined(separator: ","))
        }
        let s = spread(seedRuns), n = spread(noneRuns), c = spread(clearRuns)
        print("[repro] SEED  EOA=\(s.frames) peakdBFS=[\(s.peakDb)] cacheΔ=\(mb(seedRuns.last!.cacheAfter - seedRuns.first!.cacheBefore))MB")
        print("[repro] NONE  EOA=\(n.frames) peakdBFS=[\(n.peakDb)] cacheΔ=\(mb(noneRuns.last!.cacheAfter - noneRuns.first!.cacheBefore))MB")
        print("[repro] CLEAR EOA=\(c.frames) peakdBFS=[\(c.peakDb)] cacheΔ=\(mb(clearRuns.last!.cacheAfter - clearRuns.first!.cacheBefore))MB")
        print("[repro] WAVs: \(outDir.path)")

        // A6a refutation: with a fixed seed, every run must be bit-identical
        // regardless of how the GPU cache pool grows between runs.
        for (i, r) in seedRuns.enumerated().dropFirst() {
            XCTAssertEqual(seedRuns[0].numFrames, r.numFrames,
                           "SEED run \(i+1) EOA differs from run 1 — output depends on hidden state beyond the seed")
            XCTAssertEqual(seedRuns[0].waveSum, r.waveSum, accuracy: 1e-2,
                           "SEED run \(i+1) waveform differs from run 1 — refutes seed-determinism / implies memory pollution")
        }
    }
}
