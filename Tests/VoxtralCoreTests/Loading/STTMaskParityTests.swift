/**
 * STTMaskParityTests - K-3 parity (P-02, P-17)
 *
 * The attention masks built by the cache must not change what the STT decoder computes: last-position logits of
 * the prefill within L2 relative 1e-3 of the previous masks, greedy transcription identical, on C-court EN/FR and
 * C-moyen EN (mini-3b-8bit, `.mlx`).
 * First run (before the change) saves `.local-runs/k3_before/<clip>.safetensors`; later runs compare to it.
 * Heavy (real model): skipped unless VOXTRAL_PARITY=1 (`TEST_RUNNER_VOXTRAL_PARITY=1 xcodebuild test …`).
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class STTMaskParityTests: XCTestCase {

    private let clips = [("c_court_en", "fluxforge_short_en_6bit.wav", "en"),
                         ("c_court_fr", "fluxforge_short_fr_6bit.wav", "fr"),
                         ("c_moyen_en", "fluxforge_long_en_6bit.wav", "en")]

    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    func testPrefillLogitsAndGreedyUnchanged() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_PARITY"] == "1",
                          "Set VOXTRAL_PARITY=1 to run the K-3 parity test")
        let pipeline = VoxtralPipeline(model: .mini3b8bit, backend: .mlx)
        try await pipeline.loadModel()
        let model = try XCTUnwrap(pipeline.voxtralModel)
        let dir = root.appendingPathComponent(".local-runs/k3_before")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        var identical = 0, maxL2: Float = 0
        for (name, file, language) in clips {
            nonisolated(unsafe) var logits: MLXArray?
            model.prefillLogitsObserver = { logits = $0.asType(.float32) }
            let text = try await pipeline.transcribe(
                audio: root.appendingPathComponent("docs/examples/\(file)"), language: language)
            model.prefillLogitsObserver = nil
            let current = try XCTUnwrap(logits)
            MLX.eval(current)

            let saved = dir.appendingPathComponent("\(name).safetensors")
            guard FileManager.default.fileExists(atPath: saved.path) else {
                try MLX.save(arrays: ["logits": current], metadata: ["text": text], url: saved)
                print("[parity] \(name): saved the reference (\(text.count) chars)")
                continue
            }
            let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: saved)
            let before = try XCTUnwrap(arrays["logits"])
            let l2 = (MLX.sqrt(MLX.sum((current - before) * (current - before))) / MLX.sqrt(MLX.sum(before * before)))
                .item(Float.self)
            let same = metadata["text"] == text
            print("[parity] \(name): logits L2 rel = \(l2) ; greedy identical = \(same)")
            maxL2 = max(maxL2, l2)
            if same { identical += 1 }
            XCTAssertLessThanOrEqual(l2, 1e-3, name)
            XCTAssertTrue(same, "\(name): greedy transcription changed")
        }
        print("[parity] PARITY greedy \(identical)/\(clips.count) identiques ; logits L2 rel max=\(maxL2)")
    }
}
