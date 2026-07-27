/**
 * TTSFundamentalDeficitProbeTests — locate where a cloned voice loses its
 * fundamental.
 *
 * Fixing the enrollment high-pass recovered most of it, but ~5 dB of
 * fundamental still separates the microphone from the synthesis (measured
 * independently on the LipDub side). Two candidates, and they call for
 * different work:
 *
 *  a) the ENROLLMENT LOSS under-constrains low frequencies. The
 *     multi-resolution STFT loss runs 8 FFT sizes; at 24 kHz five of them
 *     (334, 206, 126, 76) put a 90 Hz fundamental below one bin, so it is
 *     constrained by 3 resolutions while harmonics are constrained by all 8.
 *     An optimizer trading fundamental accuracy for harmonic accuracy would
 *     show the deficit ALREADY IN THE RECONSTRUCTION — and a low-frequency
 *     weighting would fix it.
 *  b) GENERATION loses it. Then the reconstruction is faithful and the deficit
 *     appears only when the model continues the prefix on new text, which
 *     points back at the missing codec encoder — not something a loss tweak
 *     reaches.
 *
 * This probe writes the prepared reference and the reconstruction decoded from
 * the learned codes, so the two can be compared directly; measure the band
 * ratio offline. Enrollment is otherwise a black box here — `enrollVoice`
 * returns only the embedding, which cannot be decoded back to audio.
 *
 * Env: VOXTRAL_TTS_DEFICIT=1, VOXTRAL_DEFICIT_REF (raw mic recording),
 *   VOXTRAL_DEFICIT_OUT (output dir), optional VOXTRAL_DEFICIT_EPOCHS
 *   (default 3000 — enough to converge; the deficit is systematic, not a
 *   symptom of stopping early).
 */

import XCTest
import MLX
@testable import VoxtralCore

@available(macOS 14.0, *)
final class TTSFundamentalDeficitProbeTests: XCTestCase {

    func testWhereTheFundamentalIsLost() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_TTS_DEFICIT"] == "1", "Set VOXTRAL_TTS_DEFICIT=1 to run")
        guard let refPath = env["VOXTRAL_DEFICIT_REF"], let outPath = env["VOXTRAL_DEFICIT_OUT"] else {
            throw XCTSkip("Set VOXTRAL_DEFICIT_REF and VOXTRAL_DEFICIT_OUT")
        }
        let outDir = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let epochs = env["VOXTRAL_DEFICIT_EPOCHS"].flatMap { Int($0) } ?? 3000

        guard let info = VoxtralTTSRegistry.model(withId: "tts-4b-mlx") else {
            throw XCTSkip("tts-4b-mlx not in registry")
        }
        let modelDir = try await ModelDownloader.downloadTTSModel(info)
        let model = try loadVoxtralTTSModel(from: modelDir)

        var config = VoxtralVoiceEnrollment.Config()
        config.numFrames = 200          // 16 s
        config.epochs = epochs
        config.logEvery = 500
        let enroller = VoxtralVoiceEnrollment(model: model, config: config)

        // 1. The reference exactly as the optimization sees it.
        let reference = try enroller.prepareReference(url: URL(fileURLWithPath: refPath))
        try WAVWriter.write(waveform: reference, to: outDir.appendingPathComponent("prepared_reference.wav"))
        print("[deficit] prepared reference: \(reference.dim(0)) samples")

        // 2. Optimize, then decode the learned codes back to audio. This is the
        //    optimization's own target reproduction — what the loss actually
        //    achieved, with no generation involved.
        let codes = enroller.optimize(reference: reference) { p in
            if p.epoch % 500 == 0 || p.epoch == 1 {
                print("[deficit] epoch \(p.epoch)/\(epochs) loss \(String(format: "%.4f", p.totalLoss))")
            }
        }
        // `optimize` returns (T, 37); the decoder expects the batched (1, T, 37)
        // that `generate` produces.
        let batched = codes.ndim == 2 ? MLX.expandedDimensions(codes, axis: 0) : codes
        let recon = model.decodeToWaveform(batched)
        MLX.eval(recon)
        try WAVWriter.write(waveform: recon, to: outDir.appendingPathComponent("reconstruction.wav"))
        print("[deficit] reconstruction: \(recon.dim(0)) samples -> \(outDir.path)")
    }
}
