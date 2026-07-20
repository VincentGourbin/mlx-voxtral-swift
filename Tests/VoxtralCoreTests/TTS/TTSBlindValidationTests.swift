/**
 * TTSBlindValidationTests — blind A/B of A6b warm-up combinations through the
 * REAL pipeline API (VoxtralTTSPipeline.synthesize warmUpText/warmUpLeadInFrames),
 * so this also validates the shipped code path end to end.
 *
 * Files are named sample_A…F with the mapping printed only to the log (the
 * listener can't tell which is which from the audio). Multi-block enrolled
 * voice, fixed seed to isolate the warm-up choice.
 *
 * Env: VOXTRAL_TTS_REPRO=1 + _EMB (multiblock embedding) + _OUT.
 * Loads the 6-bit model from the Hub cache via the registry.
 */

import XCTest
import MLX
@testable import VoxtralCore

final class TTSBlindValidationTests: XCTestCase {

    func testBlindWarmUpCombinations() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["VOXTRAL_TTS_REPRO"] == "1",
            "Set VOXTRAL_TTS_REPRO=1 to run")
        let env = ProcessInfo.processInfo.environment
        guard let embPath = env["VOXTRAL_ENROLL_EMB"] ?? env["VOXTRAL_TTS_REPRO_EMB"],
              let outPath = env["VOXTRAL_TTS_REPRO_OUT"] else {
            throw XCTSkip("Set VOXTRAL_TTS_REPRO_EMB / _OUT")
        }
        let outDir = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: embPath))
        guard let emb = arrays["embedding"] ?? arrays.values.first else { throw XCTSkip("no embedding") }

        let text = """
        Du nouveau dans FluxForge Studio !
        Vous pouvez maintenant entrainer vos propres voix pour réaliser des doublages digne d’une qualité profesionnelle.
        La version 3.1 apporte aussi son lot d’améliorations de performances.
        """

        let pipeline = VoxtralTTSPipeline()
        guard let info = VoxtralTTSRegistry.model(withId: "tts-4b-6bit") else {
            throw XCTSkip("6-bit registry model missing")
        }
        try await pipeline.loadModel(modelInfo: info)

        // Shuffled, secret mapping (letter -> combination). Not obvious order.
        struct Combo { let letter: String; let warmUp: String?; let lead: Int; let desc: String }
        let seed: UInt64 = 1234
        let combos: [Combo] = [
            Combo(letter: "A", warmUp: "Ah ah ah ah ah ah ah.",       lead: 0, desc: "vocalise ah-ah, lead 0"),
            Combo(letter: "B", warmUp: nil,                            lead: 0, desc: "baseline, no warm-up"),
            Combo(letter: "C", warmUp: "La la la la la la la la.",     lead: 3, desc: "vocalise la-la, lead 3 (0.24s)"),
            Combo(letter: "D", warmUp: "Bonjour à toutes et à tous.",  lead: 0, desc: "verbal carrier, lead 0"),
            Combo(letter: "E", warmUp: "Mmmmmmmmmmmmmm.",              lead: 0, desc: "vocalise hum, lead 0"),
            Combo(letter: "F", warmUp: "La la la la la la la la.",     lead: 0, desc: "vocalise la-la, lead 0"),
        ]

        print("[blind] SECRET MAPPING (do not reveal to listener):")
        for c in combos { print("[blind]   sample_\(c.letter) = \(c.desc)") }

        for c in combos {
            let result = try await pipeline.synthesize(
                text: text, voiceEmbedding: emb, seed: seed,
                warmUpText: c.warmUp, warmUpLeadInFrames: c.lead)
            let url = outDir.appendingPathComponent("sample_\(c.letter).wav")
            try WAVWriter.write(waveform: result.waveform, to: url, sampleRate: result.sampleRate)
            print(String(format: "[blind] wrote sample_%@ dur=%.1fs", c.letter, result.duration))
        }
        print("[blind] done — blind samples in \(outDir.path)")
    }
}
