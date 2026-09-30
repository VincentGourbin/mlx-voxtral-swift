/**
 * CrossActorWaveformTests - K-16 (S-12, MLX-004)
 *
 * A synthesis run off the MainActor hands its waveform (an MLXArray inside the
 * `@unchecked Sendable` TTSSynthesisResult) to the MainActor, which writes the WAV.
 * 20 runs with a fixed seed: no crash, WAV files identical byte for byte. Preventive:
 * no crash was reproduced before the fix.
 *
 * Heavy (loads the TTS model): skipped unless VOXTRAL_CROSS_ACTOR=1
 * (`TEST_RUNNER_VOXTRAL_CROSS_ACTOR=1 xcodebuild test …`). Optional:
 * VOXTRAL_MODELS_DIR (models directory, downloaded there if missing),
 * VOXTRAL_TTS_MODEL (registry id, default tts-4b-4bit).
 */

import CryptoKit
import Foundation
import XCTest
@testable import VoxtralCore

final class CrossActorWaveformTests: XCTestCase {

    func testSynthesisOffMainActorConsumedOnMainActor() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_CROSS_ACTOR"] == "1",
                          "Set VOXTRAL_CROSS_ACTOR=1 to run this heavy test")

        let savedDir = ModelDownloader.customModelsDirectory
        defer { ModelDownloader.customModelsDirectory = savedDir }
        if let dir = env["VOXTRAL_MODELS_DIR"] {
            ModelDownloader.customModelsDirectory = URL(fileURLWithPath: dir)
        }
        let modelId = env["VOXTRAL_TTS_MODEL"] ?? "tts-4b-4bit"
        let modelInfo = try XCTUnwrap(VoxtralTTSRegistry.model(withId: modelId))

        let pipeline = VoxtralTTSPipeline(configuration: .init(maxFrames: 200))
        try await pipeline.loadModel(modelInfo: modelInfo)
        defer { pipeline.unload() }

        let outDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxtral-cross-actor-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outDir) }

        var reference: Data?
        var identical = 0
        for run in 1...20 {
            let result = try await Task.detached {
                try await pipeline.synthesize(text: "The waveform crosses from a worker to the main actor.",
                                              voice: .neutralFemale, seed: 42)
            }.value

            let url = outDir.appendingPathComponent("run\(run).wav")
            let bytes = try await MainActor.run { () throws -> Data in
                try WAVWriter.write(waveform: result.waveform, to: url)
                return try Data(contentsOf: url)
            }

            if let reference {
                XCTAssertEqual(bytes, reference, "run \(run) differs from run 1")
                if bytes == reference { identical += 1 }
            } else {
                reference = bytes
                identical += 1
            }
        }
        let digest = SHA256.hash(data: reference ?? Data()).map { String(format: "%02x", $0) }.joined()
        print("[cross-actor] \(identical)/20 WAV identical to run 1 (\(reference?.count ?? 0) bytes, sha256 \(digest))")
        XCTAssertEqual(identical, 20)
    }
}
