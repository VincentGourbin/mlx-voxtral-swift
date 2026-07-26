/**
 * TTSEmbeddingComparisonTests — synthesize the SAME text with two voice
 * embeddings under identical conditions (same seed, same warm-up carrier) and
 * write both WAVs, so two enrollments can be compared without sampling
 * variance confounding the result.
 *
 * Built to validate the enrollment high-pass fix (issue #45 follow-up): the
 * old complementary-FIR high-pass attenuated a male fundamental by 24–27 dB,
 * so a voice re-enrolled from the SAME reference with the fixed filter should
 * show a markedly better F0/H2 ratio. Measure the written WAVs offline.
 *
 * Env: VOXTRAL_TTS_COMPARE=1 (gate), VOXTRAL_EMB_A / VOXTRAL_EMB_B (paths to
 *   the two .safetensors), VOXTRAL_COMPARE_OUT (output dir), optional
 *   VOXTRAL_COMPARE_TEXT and VOXTRAL_COMPARE_SEED (default 42).
 */

import XCTest
import MLX
@testable import VoxtralCore

@available(macOS 14.0, *)
final class TTSEmbeddingComparisonTests: XCTestCase {

    func testCompareTwoEmbeddings() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_COMPARE"] == "1",
                          "Set VOXTRAL_TTS_COMPARE=1 to run this heavy comparison")
        guard let embA = env["VOXTRAL_EMB_A"], let embB = env["VOXTRAL_EMB_B"],
              let outPath = env["VOXTRAL_COMPARE_OUT"] else {
            throw XCTSkip("Set VOXTRAL_EMB_A / VOXTRAL_EMB_B / VOXTRAL_COMPARE_OUT")
        }
        let outDir = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        let text = env["VOXTRAL_COMPARE_TEXT"] ?? "Fluxforge Studio transforme votre Mac en un studio de création IA complet. Générez des images et des vidéos de haute qualité à partir de texte, entraînez vos propres modèles personnalisés, et gérez votre bibliothèque créative — le tout en local sur votre Apple Silicon, sans cloud ni abonnement."
        let seed = env["VOXTRAL_COMPARE_SEED"].flatMap { UInt64($0) } ?? 42

        // bf16 by default: issue #45 item 3 measured q6 dropping words with
        // enrolled voices, and the enrollment itself runs against bf16.
        let modelId = env["VOXTRAL_COMPARE_MODEL"] ?? "tts-4b-mlx"
        guard let modelInfo = VoxtralTTSRegistry.model(withId: modelId) else {
            throw XCTSkip("Unknown TTS model id: \(modelId)")
        }
        let pipeline = VoxtralTTSPipeline()
        try await pipeline.loadModel(modelInfo: modelInfo)

        func load(_ path: String) throws -> MLXArray {
            let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            guard let e = arrays["embedding"] ?? arrays.values.first else {
                throw XCTSkip("No embedding array in \(path)")
            }
            return e
        }

        // Frames of the carrier's terminal pause kept before the content. With
        // 0 (a tight cut) the cut lands on the 80 ms frame where speech
        // resumes, so a word starting mid-frame loses its attack — heard as a
        // clipped first word ("Fluxforge" → "…orge").
        let leadIn = env["VOXTRAL_COMPARE_LEADIN"].flatMap { Int($0) } ?? 2

        for (label, path) in [("A", embA), ("B", embB)] {
            let embedding = try load(path)
            let result = try await pipeline.synthesize(
                text: text, voiceEmbedding: embedding, seed: seed,
                warmUpText: VoxtralTTSPipeline.recommendedWarmUpVocalise,
                warmUpLeadInFrames: leadIn)
            let url = outDir.appendingPathComponent("compare_\(label).wav")
            try WAVWriter.write(waveform: result.waveform, to: url)
            let dur = Double(result.waveform.dim(0)) / Double(pipeline.sampleRate)
            print(String(format: "[compare] %@ (%@): %d frames, %.2f s -> %@",
                         label, URL(fileURLWithPath: path).lastPathComponent,
                         result.numFrames, dur, url.path))
        }
    }
}
