/**
 * TTSConsecutiveSynthesisReproTests — investigation harness for the
 * Fluxforge A6 report: consecutive syntheses with the same enrolled voice
 * degrade cumulatively (3rd preview unintelligible, max level −17 dB vs
 * −5 dB healthy; reloading the model resets it).
 *
 * Two hypotheses are separated:
 *  1. App path — `generate` WITHOUT prefix cache (what Fluxforge previews
 *     actually call). With a fixed RNG seed, runs must be bit-identical;
 *     any drift proves persistent mutable state in the model.
 *  2. A6's claimed mechanism — the memoized voice-prefix KV cache being
 *     mutated by generation. State checksums before/after two prefix-cache
 *     generations must be identical.
 *
 * Heavy (loads the real 4B model): skipped unless VOXTRAL_TTS_REPRO=1.
 * Optional VOXTRAL_TTS_REPRO_EMB points at an enrolled-voice .safetensors.
 */

import XCTest
import MLX
import MLXLMCommon
import MLXRandom
@testable import VoxtralCore

final class TTSConsecutiveSynthesisReproTests: XCTestCase {

    private struct RunStats {
        let maxAbs: Float
        let rms: Float
        let numFrames: Int
        let samples: Int
        var maxDb: Float { 20 * log10(max(maxAbs, 1e-9)) }
        var rmsDb: Float { 20 * log10(max(rms, 1e-9)) }
    }

    private func stats(_ waveform: MLXArray, numFrames: Int) -> RunStats {
        let absW = MLX.abs(waveform.asType(.float32))
        let maxAbs = absW.max().item(Float.self)
        let rms = MLX.sqrt(MLX.mean(waveform.asType(.float32) * waveform.asType(.float32))).item(Float.self)
        return RunStats(maxAbs: maxAbs, rms: rms, numFrames: numFrames, samples: waveform.dim(0))
    }

    private func cacheChecksums(_ cache: [any KVCache]) -> [(offset: Int, sums: [Float])] {
        cache.map { c in
            (c.offset, c.state.map { $0.asType(.float32).sum().item(Float.self) })
        }
    }

    func testConsecutiveSynthesesWithSameVoice() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["VOXTRAL_TTS_REPRO"] == "1",
            "Set VOXTRAL_TTS_REPRO=1 to run this heavy repro test")

        let modelDir: URL
        if let override = ProcessInfo.processInfo.environment["VOXTRAL_TTS_REPRO_MODEL"] {
            modelDir = URL(fileURLWithPath: override)
        } else if let found = ModelDownloader.findTTSModelPath(for: VoxtralTTSRegistry.defaultModel) {
            modelDir = found
        } else {
            throw XCTSkip("Default TTS model not downloaded")
        }
        guard let embPath = ProcessInfo.processInfo.environment["VOXTRAL_TTS_REPRO_EMB"] else {
            throw XCTSkip("Set VOXTRAL_TTS_REPRO_EMB to an enrolled-voice .safetensors path")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: embPath))
        guard let voiceEmb = arrays["embedding"] ?? arrays.values.first else {
            throw XCTSkip("No embedding in \(embPath)")
        }
        print("[repro] voice embedding shape: \(voiceEmb.shape)")

        let model = try loadVoxtralTTSModel(from: modelDir)
        let tokenizer = TekkenTokenizer(modelPath: modelDir.path)

        let text = "Bonjour, ceci est un test de synthèse vocale avec une voix personnalisée pour vérifier la stabilité."
        let outDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["VOXTRAL_TTS_REPRO_OUT"]
            ?? NSTemporaryDirectory()).appendingPathComponent("tts_repro")
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        // ---- Part 1: app path (no prefix cache), fixed seed each run ----
        var appRuns: [RunStats] = []
        var appWaveSums: [Float] = []
        for run in 1...3 {
            MLXRandom.seed(42)
            let (codes, numFrames, _) = model.generate(
                text: text, voiceEmbedding: voiceEmb, tokenizer: tokenizer,
                maxTokens: 1500, sanitize: true)
            let waveform = model.decodeToWaveform(codes)
            MLX.eval(waveform)
            let s = stats(waveform, numFrames: numFrames)
            appRuns.append(s)
            appWaveSums.append(waveform.asType(.float32).sum().item(Float.self))
            try WAVWriter.write(waveform: waveform, to: outDir.appendingPathComponent("app_run\(run).wav"))
            print(String(format: "[repro] APP run %d: frames=%d samples=%d max=%.4f (%.1f dB) rms=%.5f (%.1f dB) waveSum=%.4f",
                         run, s.numFrames, s.samples, s.maxAbs, s.maxDb, s.rms, s.rmsDb, appWaveSums.last!))
        }

        // ---- Part 2: prefix-cache path (A6's claimed mechanism) ----
        let (prefixCache, prefixLen) = model.precomputeVoicePrefixCache(voiceEmbedding: voiceEmb)
        let before = cacheChecksums(prefixCache)
        print("[repro] prefix len=\(prefixLen), layers=\(prefixCache.count)")

        var pfxRuns: [RunStats] = []
        for run in 1...3 {
            MLXRandom.seed(42)
            let (codes, numFrames, _) = model.generate(
                text: text, voiceEmbedding: voiceEmb, tokenizer: tokenizer,
                maxTokens: 1500, sanitize: true,
                prefixCache: prefixCache, prefixLen: prefixLen)
            let waveform = model.decodeToWaveform(codes)
            MLX.eval(waveform)
            let s = stats(waveform, numFrames: numFrames)
            pfxRuns.append(s)
            try WAVWriter.write(waveform: waveform, to: outDir.appendingPathComponent("pfx_run\(run).wav"))
            print(String(format: "[repro] PFX run %d: frames=%d samples=%d max=%.4f (%.1f dB) rms=%.5f (%.1f dB)",
                         run, s.numFrames, s.samples, s.maxAbs, s.maxDb, s.rms, s.rmsDb))

            let after = cacheChecksums(prefixCache)
            var mutated = 0
            for (b, a) in zip(before, after) where b.offset != a.offset || b.sums != a.sums {
                mutated += 1
            }
            print("[repro] PFX run \(run): mutated prefix layers = \(mutated)/\(prefixCache.count)")
            XCTAssertEqual(mutated, 0, "Prefix KV cache was mutated by generation (run \(run))")
        }

        // ---- Verdicts ----
        print("[repro] WAVs written to \(outDir.path)")
        for (i, s) in appRuns.enumerated().dropFirst() {
            XCTAssertEqual(appRuns[0].numFrames, s.numFrames,
                           "APP path: run \(i+1) frame count differs from run 1 (seeded identically)")
            XCTAssertEqual(appWaveSums[0], appWaveSums[i], accuracy: 1e-2,
                           "APP path: run \(i+1) waveform differs from run 1 (seeded identically)")
        }
        for (i, s) in pfxRuns.enumerated().dropFirst() {
            XCTAssertEqual(pfxRuns[0].numFrames, s.numFrames,
                           "PFX path: run \(i+1) frame count differs from run 1 (seeded identically)")
        }
    }
}
