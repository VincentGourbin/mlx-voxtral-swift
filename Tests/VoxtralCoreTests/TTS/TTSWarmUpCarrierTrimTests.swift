/**
 * TTSWarmUpCarrierTrimTests — guards the warm-up carrier trim on the BATCH
 * `synthesize(...)` path against BOTH failure directions.
 *
 * The trim can fail two ways, and each hides the other:
 *  - cut too LATE / not at all → the vocalise leaks before the content
 *    (issue #45 item 2), detectable as the carrier's true-silence terminal
 *    pause surviving in the output's first second;
 *  - cut too EARLY → real speech is eaten. With an enrolled voice the carrier
 *    renders much quieter than the content, so the default peak-relative
 *    threshold lands above the carrier: the leading-silence scan swallows the
 *    carrier and its pause, and the first gap found is the pause after
 *    sentence one. Measured on a 2-sentence text: the whole first sentence
 *    disappeared (18.9 s → 11.8 s) while the output still looked "clean".
 *
 * Comparing against a no-warm-up run at the same seed catches the second case,
 * which a silence check alone cannot see.
 *
 * Heavy (loads the real 4B model): skipped unless VOXTRAL_TTS_CARRIER=1.
 * VOXTRAL_TTS_REPRO_EMB optionally points at an enrolled-voice .safetensors
 * (the enrolled case is the one that regressed; a preset is used otherwise).
 */

import XCTest
import MLX
@testable import VoxtralCore

@available(macOS 14.0, *)
final class TTSWarmUpCarrierTrimTests: XCTestCase {

    func testCarrierTrimKeepsAllContent() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_CARRIER"] == "1",
                          "Set VOXTRAL_TTS_CARRIER=1 to run this heavy test")

        let modelId = env["VOXTRAL_TTS_CARRIER_MODEL"] ?? "tts-4b-mlx"
        guard let modelInfo = VoxtralTTSRegistry.model(withId: modelId) else {
            throw XCTSkip("Unknown TTS model id: \(modelId)")
        }
        let pipeline = VoxtralTTSPipeline()
        try await pipeline.loadModel(modelInfo: modelInfo)

        // Two sentences: the failure ate everything before the second one.
        let text = "Fluxforge Studio transforme votre Mac en un studio de création IA complet. Générez des images et des vidéos de haute qualité à partir de texte, entraînez vos propres modèles personnalisés."
        let seed: UInt64 = 42

        // `warmUpText` is only exposed on the embedding overload, and the
        // regression is specific to enrolled voices anyway (the carrier renders
        // quiet relative to the content), so an embedding is required here.
        guard let embPath = env["VOXTRAL_TTS_REPRO_EMB"] else {
            throw XCTSkip("Set VOXTRAL_TTS_REPRO_EMB to an enrolled-voice .safetensors path")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: embPath))
        guard let embedding = arrays["embedding"] ?? arrays.values.first else {
            throw XCTSkip("No embedding array in \(embPath)")
        }

        func run(warmUp: String?) async throws -> MLXArray {
            try await pipeline.synthesize(
                text: text, voiceEmbedding: embedding, seed: seed, warmUpText: warmUp
            ).waveform
        }

        let plain = try await run(warmUp: nil)
        let warmed = try await run(warmUp: VoxtralTTSPipeline.recommendedWarmUpVocalise)
        let sr = Double(pipeline.sampleRate)
        print(String(format: "[carrier-trim] no-warm-up %.2f s, warm-up %.2f s",
                     Double(plain.dim(0)) / sr, Double(warmed.dim(0)) / sr))

        // 1. Content must survive. The carrier is trimmed off, so the warm-up
        //    run is naturally a touch shorter — but losing a whole sentence
        //    shows up as a large deficit, well past this margin.
        XCTAssertGreaterThan(
            Double(warmed.dim(0)), 0.85 * Double(plain.dim(0)),
            "Warm-up run is far shorter than the plain run — the carrier trim ate real speech")

        // 2. The carrier itself must be gone: its terminal pause is true
        //    digital silence, so no such run may remain in the first second.
        let frame = 1920
        let lead = warmed.asType(.float32).asArray(Float.self)
        var run = 0, maxRun = 0
        for i in 0 ..< min(13, lead.count / frame) {
            let seg = lead[(i * frame) ..< min((i + 1) * frame, lead.count)]
            let rms = (seg.reduce(0) { $0 + $1 * $1 } / Float(seg.count)).squareRoot()
            if rms < 4e-4 { run += 1; maxRun = max(maxRun, run) } else { run = 0 }
        }
        XCTAssertLessThan(maxRun, 3,
            "A \(maxRun)-frame true-silence run survives in the first second — carrier not trimmed")
    }
}
