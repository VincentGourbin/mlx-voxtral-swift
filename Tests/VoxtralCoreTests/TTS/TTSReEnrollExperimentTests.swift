/**
 * TTSReEnrollExperimentTests — option-2 lever 1: re-enroll with tweaked
 * settings (no algorithm change) and measure whether the enrolled-voice
 * first-sentence / energy instability improves.
 *
 * Enrolls from a reference recording with env-configurable settings, saves the
 * embedding, then synthesizes the A6 test text (seed 1234) and prints a coarse
 * RMS energy envelope so runs can be compared against the baseline.
 *
 * Env: VOXTRAL_TTS_REPRO=1 + _MODEL + VOXTRAL_ENROLL_REF + _OUT, optional
 *   VOXTRAL_ENROLL_FRAMES (default 100), VOXTRAL_ENROLL_EPOCHS (default 5000),
 *   VOXTRAL_ENROLL_RECON / _PERCEPT / _MEL (loss weights), VOXTRAL_ENROLL_TAG.
 */

import XCTest
import MLX
@testable import VoxtralCore

@available(macOS 14.0, *)
final class TTSReEnrollExperimentTests: XCTestCase {

    func testReEnrollAndSynthesize() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["VOXTRAL_TTS_REPRO"] == "1",
            "Set VOXTRAL_TTS_REPRO=1 to run")
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["VOXTRAL_TTS_REPRO_MODEL"],
              let refPath = env["VOXTRAL_ENROLL_REF"],
              let outPath = env["VOXTRAL_ENROLL_OUT"] else {
            throw XCTSkip("Set VOXTRAL_TTS_REPRO_MODEL / VOXTRAL_ENROLL_REF / VOXTRAL_ENROLL_OUT")
        }
        let modelDir = URL(fileURLWithPath: modelPath)
        let outDir = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        func f(_ k: String) -> Float? { env[k].flatMap { Float($0) } }
        func i(_ k: String) -> Int? { env[k].flatMap { Int($0) } }
        let tag = env["VOXTRAL_ENROLL_TAG"] ?? "exp"

        var cfg = VoxtralVoiceEnrollment.Config()
        cfg.numFrames = i("VOXTRAL_ENROLL_FRAMES") ?? cfg.numFrames
        cfg.epochs = i("VOXTRAL_ENROLL_EPOCHS") ?? cfg.epochs
        cfg.reconstructionWeight = f("VOXTRAL_ENROLL_RECON") ?? cfg.reconstructionWeight
        cfg.perceptualWeight = f("VOXTRAL_ENROLL_PERCEPT") ?? cfg.perceptualWeight
        cfg.melWeight = f("VOXTRAL_ENROLL_MEL") ?? cfg.melWeight
        cfg.logEvery = max(1, cfg.epochs / 20)

        print("[enroll] tag=\(tag) frames=\(cfg.numFrames) epochs=\(cfg.epochs) " +
              "recon=\(cfg.reconstructionWeight) percept=\(cfg.perceptualWeight) mel=\(cfg.melWeight)")

        let model = try loadVoxtralTTSModel(from: modelDir)
        let tokenizer = TekkenTokenizer(modelPath: modelDir.path)

        // --- Enroll ---
        let enroller = VoxtralVoiceEnrollment(model: model, config: cfg)
        let reference = try enroller.prepareReference(url: URL(fileURLWithPath: refPath))
        let codes = enroller.optimize(reference: reference) { p in
            if p.epoch % cfg.logEvery == 0 {
                print(String(format: "[enroll] epoch %d/%d total=%.4f recon=%.4f",
                             p.epoch, cfg.epochs, p.totalLoss, p.reconLoss))
            }
        }
        let emb = enroller.codesToVoiceEmbedding(codes)
        let embURL = outDir.appendingPathComponent("enrolled_\(tag).safetensors")
        try MLX.save(arrays: ["embedding": emb], url: embURL)
        print("[enroll] saved \(embURL.lastPathComponent) shape=\(emb.shape)")

        // --- Synthesize the A6 test text with the fresh embedding ---
        let text = """
        Du nouveau dans FluxForge Studio !
        Vous pouvez maintenant entrainer vos propres voix pour réaliser des doublages digne d’une qualité profesionnelle.
        La version 3.1 apporte aussi son lot d’améliorations de performances.
        """
        let (gcodes, _, _) = model.generate(
            text: text, voiceEmbedding: emb, tokenizer: tokenizer,
            maxTokens: 1500, sanitize: true, seed: 1234)
        let wav = model.decodeToWaveform(gcodes)
        MLX.eval(wav)
        let wavURL = outDir.appendingPathComponent("synth_\(tag).wav")
        try WAVWriter.write(waveform: wav, to: wavURL)

        // Coarse 0.5 s RMS envelope for energy-stability comparison.
        let sr = 24000, win = sr / 2
        let f32 = wav.asType(.float32)
        let total = f32.dim(0)
        var env0: [String] = []
        var s = 0
        while s + win / 2 < total {
            let e = min(s + win, total)
            let r = MLX.sqrt(MLX.mean(f32[s..<e] * f32[s..<e])).item(Float.self)
            env0.append(String(format: "%.0f", 20 * log10(max(r, 1e-9))))
            s += win
        }
        print("[enroll] synth dur=\(String(format: "%.1f", Double(total) / Double(sr)))s " +
              "envelope dBFS: \(env0.joined(separator: " "))")
        print("[enroll] done — \(embURL.path), \(wavURL.path)")
    }
}
