/**
 * TTSCarrierCodeProbeTests — probe whether the warm-up carrier's boundary is
 * visible in CODE space (per-frame semantic code) rather than only in energy.
 *
 * The energy heuristic in trimLeadingCarrier is demonstrably unreliable: on a
 * seeded sweep it leaked the vocalise into the output on 2 of 3 seeds and, on
 * another, cut a whole sentence away. Issue #45 suggests a "token-aligned" cut,
 * but text tokens carry no audio-frame alignment — the whole prompt is
 * prefilled at once and the model paces the frames itself.
 *
 * What DOES exist per frame is the semantic code (`codes[0, 0]`), the same
 * signal the generator uses to spot end-of-audio (`semanticCode <= 1`). This
 * probe dumps, for each frame, the semantic code alongside the frame's decoded
 * energy, for (a) the carrier alone and (b) carrier + content, so the boundary
 * can be located and any code-space marker identified.
 *
 * Env: VOXTRAL_TTS_PROBE=1, VOXTRAL_TTS_REPRO_EMB (enrolled voice),
 *   optional VOXTRAL_TTS_PROBE_SEED (default 777 — a seed that leaked).
 */

import XCTest
import MLX
@testable import VoxtralCore

@available(macOS 14.0, *)
final class TTSCarrierCodeProbeTests: XCTestCase {

    func testProbeCarrierBoundaryInCodeSpace() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_PROBE"] == "1", "Set VOXTRAL_TTS_PROBE=1 to run")
        guard let embPath = env["VOXTRAL_TTS_REPRO_EMB"] else {
            throw XCTSkip("Set VOXTRAL_TTS_REPRO_EMB")
        }
        let seed = env["VOXTRAL_TTS_PROBE_SEED"].flatMap { UInt64($0) } ?? 777

        guard let modelInfo = VoxtralTTSRegistry.model(withId: "tts-4b-mlx") else {
            throw XCTSkip("tts-4b-mlx not in registry")
        }
        let modelDir = try await ModelDownloader.downloadTTSModel(modelInfo)
        let model = try loadVoxtralTTSModel(from: modelDir)
        let tokenizer = TekkenTokenizer(modelPath: modelDir.path)

        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: embPath))
        guard let emb = arrays["embedding"] ?? arrays.values.first else {
            throw XCTSkip("No embedding in \(embPath)")
        }

        let carrier = VoxtralTTSPipeline.recommendedWarmUpVocalise
        let content = "Fluxforge Studio transforme votre Mac en un studio de création IA complet."

        func dump(_ text: String, label: String) {
            MLXRandom.seed(seed)
            let (codes, numFrames, _) = model.generate(
                text: text, voiceEmbedding: emb, tokenizer: tokenizer,
                maxTokens: 600, sanitize: true, seed: seed)
            guard numFrames > 0 else { print("[probe] \(label): no frames"); return }
            let wav = model.decodeToWaveform(codes)
            MLX.eval(wav)
            let samples = wav.asType(.float32).asArray(Float.self)
            let sem = codes[0, 0..., 0].asArray(Int32.self)
            print("[probe] \(label): \(numFrames) frames")
            let frame = 1920
            for i in 0 ..< numFrames {
                let lo = i * frame, hi = min(lo + frame, samples.count)
                guard lo < hi else { break }
                var acc: Float = 0
                for s in samples[lo ..< hi] { acc += s * s }
                let db = 20 * log10(max((acc / Float(hi - lo)).squareRoot(), 1e-9))
                print(String(format: "[probe] %@ f%03d sem=%6d  %6.1f dB", label, i, sem[i], db))
            }
        }

        dump(carrier, label: "CARRIER-ONLY")
        dump(carrier + " " + content, label: "CARRIER+CONTENT")
    }
}
