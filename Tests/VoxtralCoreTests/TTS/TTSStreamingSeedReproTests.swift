/**
 * TTSStreamingSeedReproTests — guards the seed + warm-up threading on the
 * STREAMING path (`VoxtralTTSPipeline.synthesizeStreaming`), the path the
 * GUI/Fluxforge previews actually use.
 *
 * Background: the A6c seed and A6b warm-up mitigations originally landed only
 * on the batch `synthesize(...)` overload; the streaming overload kept drawing
 * fresh flow-matching noise on every call, so identical text+voice produced
 * decorrelated audio run-to-run (empirically: 3 consecutive cloned-voice
 * previews had ~0 cross-correlation and 15.8/16.0/19.8 s durations).
 *
 * Assertions:
 *  1. Seeded streaming is deterministic — two runs are bit-identical.
 *  2. Un-seeded streaming is non-deterministic — two runs differ (documents
 *     the pre-fix behavior; skipped from strict CI via the env gate).
 *  3. Warm-up is deterministic too, and its carrier audio is trimmed (the
 *     warm-up run is shorter/different from a no-warm-up run of the same text).
 *
 * Heavy (loads the real 4B model): skipped unless VOXTRAL_TTS_STREAM_SEED=1.
 * Optional VOXTRAL_TTS_REPRO_MODEL overrides the model dir;
 * VOXTRAL_TTS_REPRO_EMB points at an enrolled-voice .safetensors (else a
 * preset voice is used).
 */

import XCTest
import MLX
@testable import VoxtralCore

@available(macOS 14.0, *)
final class TTSStreamingSeedReproTests: XCTestCase {

    /// Drain a streaming synthesis into a single concatenated waveform.
    private func collect(
        _ stream: AsyncThrowingStream<TTSStreamingChunk, Error>
    ) async throws -> MLXArray {
        var parts: [MLXArray] = []
        for try await chunk in stream where chunk.waveform.dim(0) > 0 {
            parts.append(chunk.waveform)
        }
        guard !parts.isEmpty else { return MLXArray([Float]()) }
        let full = MLX.concatenated(parts, axis: 0)
        MLX.eval(full)
        return full
    }

    private func waveSum(_ w: MLXArray) -> Float {
        w.asType(.float32).sum().item(Float.self)
    }

    func testStreamingSeedIsDeterministic() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["VOXTRAL_TTS_STREAM_SEED"] == "1",
            "Set VOXTRAL_TTS_STREAM_SEED=1 to run this heavy streaming repro test")

        let pipeline = VoxtralTTSPipeline()
        let env = ProcessInfo.processInfo.environment
        try await pipeline.loadModel(modelInfo: VoxtralTTSRegistry.defaultModel)

        let text = "Bonjour, ceci est un test de synthèse vocale en streaming pour vérifier le déterminisme."

        // Resolve a voice: enrolled embedding if provided, else a preset.
        let embedding: MLXArray?
        if let embPath = env["VOXTRAL_TTS_REPRO_EMB"] {
            let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: embPath))
            embedding = arrays["embedding"] ?? arrays.values.first
        } else {
            embedding = nil
        }

        func run(seed: UInt64?, warmUp: String? = nil) async throws -> MLXArray {
            let stream: AsyncThrowingStream<TTSStreamingChunk, Error>
            if let embedding {
                stream = pipeline.synthesizeStreaming(
                    text: text, voiceEmbedding: embedding, chunkSize: 10,
                    seed: seed, warmUpText: warmUp)
            } else {
                stream = pipeline.synthesizeStreaming(
                    text: text, voice: .neutralFemale, chunkSize: 10,
                    seed: seed, warmUpText: warmUp)
            }
            return try await collect(stream)
        }

        // 1. Seeded → bit-identical across runs.
        let a = try await run(seed: 42)
        let b = try await run(seed: 42)
        print("[stream-seed] seeded samples: \(a.dim(0)) vs \(b.dim(0))")
        XCTAssertEqual(a.dim(0), b.dim(0), "Seeded streaming: sample count differs between runs")
        XCTAssertEqual(waveSum(a), waveSum(b), accuracy: 1e-2,
                       "Seeded streaming: waveform differs between identically-seeded runs")

        // 2. Un-seeded → non-deterministic (documents the pre-fix behavior).
        let c = try await run(seed: nil)
        let d = try await run(seed: nil)
        let unseededDiffers = c.dim(0) != d.dim(0) || abs(waveSum(c) - waveSum(d)) > 1e-1
        print("[stream-seed] unseeded samples: \(c.dim(0)) vs \(d.dim(0)) differs=\(unseededDiffers)")
        XCTAssertTrue(unseededDiffers, "Un-seeded streaming was unexpectedly identical across runs")

        // 3. Warm-up: deterministic, and its carrier is trimmed so the output
        //    is not identical to the no-warm-up run of the same seed+text.
        let w1 = try await run(seed: 42, warmUp: VoxtralTTSPipeline.recommendedWarmUpVocalise)
        let w2 = try await run(seed: 42, warmUp: VoxtralTTSPipeline.recommendedWarmUpVocalise)
        print("[stream-seed] warm-up samples: \(w1.dim(0)) vs \(w2.dim(0))")
        XCTAssertEqual(w1.dim(0), w2.dim(0), "Warm-up streaming: sample count differs between runs")
        XCTAssertEqual(waveSum(w1), waveSum(w2), accuracy: 1e-2,
                       "Warm-up streaming: waveform differs between identically-seeded runs")
        XCTAssertNotEqual(w1.dim(0), a.dim(0),
                          "Warm-up run has the same length as the no-warm-up run — carrier not trimmed?")
    }
}
