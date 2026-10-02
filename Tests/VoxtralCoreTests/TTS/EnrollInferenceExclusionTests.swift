/**
 * EnrollInferenceExclusionTests - K-11 (A-01, piège 20)
 *
 * An enrollment (gradient, vjp) and a streaming synthesis (compiled kernels) on the same pipeline
 * must never run at the same time: mlx-swift 0.31.6 takes its compile and vjp locks in opposite
 * orders (ABBA deadlock). 20 runs of `enrollVoice` (50 epochs) ∥ `synthesizeStreaming`: no hang
 * (120 s per run), the synthesis is refused `busy` in < 1 s or runs after the enrollment.
 *
 * Heavy (real TTS 4-bit model): skipped unless VOXTRAL_ENROLL_EXCLUSION=1
 * (`TEST_RUNNER_VOXTRAL_ENROLL_EXCLUSION=1 xcodebuild test …`); VOXTRAL_MODELS_DIR optional.
 * Uses only APIs that predate K-11 so it compiles, and fails, without the fix.
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class EnrollInferenceExclusionTests: XCTestCase {

    /// Wall-clock marks shared between the enrollment thread and the test.
    private final class Timeline: @unchecked Sendable {
        let lock = NSLock()
        var enrollStarted = false
        var enrollEnd: Date?
        var firstChunk: Date?
    }

    func testEnrollmentAndStreamingNeverOverlap() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["VOXTRAL_ENROLL_EXCLUSION"] == "1", "Set VOXTRAL_ENROLL_EXCLUSION=1 to run")
        let saved = VoxtralModelDownloader.customModelsDirectory
        defer { VoxtralModelDownloader.customModelsDirectory = saved }
        if let dir = env["VOXTRAL_MODELS_DIR"] { VoxtralModelDownloader.customModelsDirectory = URL(fileURLWithPath: dir) }

        let pipeline = VoxtralTTSPipeline(configuration: .init(maxFrames: 60))
        try await pipeline.loadModel(modelInfo: XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-4bit")))
        let reference = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("docs/examples/clone_en.wav")
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral-exclusion-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: out) }

        final class Box: @unchecked Sendable { let p: VoxtralTTSPipeline; init(_ p: VoxtralTTSPipeline) { self.p = p } }
        let box = Box(pipeline)
        var passed = 0
        var refusals: [Double] = []

        for run in 1 ... 20 {
            let timeline = Timeline()
            let enrollDone = expectation(description: "enrollment \(run)")
            Thread.detachNewThread {
                var config = VoxtralVoiceEnrollment.Config()
                config.numFrames = 50
                config.epochs = 50
                config.logEvery = 1
                _ = try? box.p.enrollVoice(referenceURL: reference, outputURL: out, config: config, progress: { _ in
                    timeline.lock.withLock { timeline.enrollStarted = true }
                })
                timeline.lock.withLock { timeline.enrollEnd = Date() }
                enrollDone.fulfill()
            }
            // Start the synthesis once the enrollment loop is running
            while !timeline.lock.withLock({ timeline.enrollStarted }) {
                if timeline.lock.withLock({ timeline.enrollEnd }) != nil { break }
                try await Task.sleep(nanoseconds: 20_000_000)
            }

            let synthDone = expectation(description: "synthesis \(run)")
            let callTime = Date()
            final class Outcome: @unchecked Sendable { var error: Error?; var errorAt: Date? }
            let outcome = Outcome()
            Task.detached {
                let voice = MLXArray.zeros([4, 3072], dtype: .bfloat16)
                do {
                    for try await _ in box.p.synthesizeStreaming(text: "Hello there.", voiceEmbedding: voice) {
                        timeline.lock.withLock { if timeline.firstChunk == nil { timeline.firstChunk = Date() } }
                    }
                } catch {
                    outcome.error = error
                    outcome.errorAt = Date()
                }
                synthDone.fulfill()
            }

            let result = await XCTWaiter().fulfillment(of: [enrollDone, synthDone], timeout: 120)
            guard result == .completed else {
                XCTFail("run \(run): no completion within 120 s (deadlock)")
                break
            }
            let (enrollEnd, firstChunk) = timeline.lock.withLock { (timeline.enrollEnd, timeline.firstChunk) }
            if let error = outcome.error, "\(error)".contains("busy"), let at = outcome.errorAt {
                let ms = at.timeIntervalSince(callTime) * 1000
                refusals.append(ms)
                XCTAssertLessThan(ms, 1000, "run \(run): busy refusal took \(ms) ms")
                passed += 1
            } else if let firstChunk, let enrollEnd, firstChunk < enrollEnd {
                XCTFail("run \(run): overlap — the synthesis produced audio during the enrollment")
                break
            } else if outcome.error == nil {
                passed += 1  // ran after the enrollment
            } else {
                XCTFail("run \(run): unexpected error \(outcome.error!)")
            }
        }
        let worst = refusals.max() ?? 0
        print("[enroll-exclusion] \(passed)/20 OK, refus busy max \(Int(worst)) ms")
        XCTAssertEqual(passed, 20)
    }
}
