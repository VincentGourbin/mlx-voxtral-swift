/**
 * EnrollmentReproTests - K-26 (A-07, A-08, A-09)
 *
 * Voice enrollment is reproducible and resumable: same seed → same codes, another seed → other codes; a run
 * stopped and resumed from its checkpoint ends with the codes of the uninterrupted run, bit for bit. The
 * divergence guard (#44) and the cancellation are tested: NaN at epoch 0 → EnrollmentDivergedError, NaN at
 * epoch k → codes of the best step, cancellation → CancellationError.
 *
 * Short runs (2 s reference, 12 epochs) through the real frozen codec of tts-4b-4bit.
 * Heavy: skipped unless VOXTRAL_ENROLL_REPRO=1 (`TEST_RUNNER_VOXTRAL_ENROLL_REPRO=1 xcodebuild test …`).
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class EnrollmentReproTests: XCTestCase {

    nonisolated(unsafe) private static var model: VoxtralTTSModel?

    private var sandbox: URL!

    override func setUp() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_ENROLL_REPRO"] == "1",
                          "Set VOXTRAL_ENROLL_REPRO=1 to run the enrollment reproducibility tests")
        if Self.model == nil {
            let pipeline = VoxtralTTSPipeline()
            try await pipeline.loadModel(modelInfo: try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-4bit")))
            Self.model = pipeline.ttsModel
        }
        sandbox = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral-enroll-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: sandbox, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let sandbox { try? FileManager.default.removeItem(at: sandbox) }
    }

    private func config(seed: UInt64?, checkpoint: URL? = nil) -> VoxtralVoiceEnrollment.Config {
        var config = VoxtralVoiceEnrollment.Config()
        config.numFrames = 25  // 2 s
        config.epochs = 12
        config.seed = seed
        config.checkpointURL = checkpoint
        config.checkpointEvery = 4
        return config
    }

    /// A deterministic 2 s "voice": two harmonics with a slow envelope, 24 kHz
    private func reference(frames: Int) -> MLXArray {
        let n = frames * VoxtralVoiceEnrollment.samplesPerFrame
        let samples = (0 ..< n).map { i -> Float in
            let t = Float(i) / 24_000
            return 0.3 * (0.6 + 0.4 * sin(2 * .pi * 3 * t)) * (sin(2 * .pi * 180 * t) + 0.5 * sin(2 * .pi * 360 * t))
        }
        return MLXArray(samples)
    }

    private func enroller(_ config: VoxtralVoiceEnrollment.Config) throws -> VoxtralVoiceEnrollment {
        VoxtralVoiceEnrollment(model: try XCTUnwrap(Self.model), config: config)
    }

    private func run(_ config: VoxtralVoiceEnrollment.Config, stopAt: Int? = nil,
                     setUp: (VoxtralVoiceEnrollment) -> Void = { _ in }) throws -> MLXArray {
        let enroller = try enroller(config)
        setUp(enroller)
        var polls = 0
        return try enroller.optimize(reference: reference(frames: config.numFrames), shouldContinue: {
            defer { polls += 1 }
            return stopAt.map { polls < $0 } ?? true
        })
    }

    private func identical(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && MLX.arrayEqual(a, b).item(Bool.self)
    }

    func testSameSeedSameCodesOtherSeedOtherCodes() throws {
        let a = try run(config(seed: 7)), b = try run(config(seed: 7)), c = try run(config(seed: 8))
        print("[enroll] SEED codes \(a.shape) seed 7 ×2 identical=\(identical(a, b)) ; seed 8 identical=\(identical(a, c))")
        XCTAssertEqual(a.shape, [25, 37])
        XCTAssertTrue(identical(a, b), "same seed must give the same codes")
        XCTAssertFalse(identical(a, c), "another seed must give other codes")
    }

    func testStopAndResumeIsBitIdentical() throws {
        let continuous = try run(config(seed: 7, checkpoint: sandbox.appendingPathComponent("continuous.safetensors")))
        let url = sandbox.appendingPathComponent("resumed.safetensors")
        XCTAssertThrowsError(try run(config(seed: 7, checkpoint: url), stopAt: 6)) { error in
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "a cancelled run leaves its checkpoint")
        let resumed = try run(config(seed: 7, checkpoint: url))
        print("[enroll] RESUME stop 6/12 then resume: identical=\(identical(continuous, resumed))")
        XCTAssertTrue(identical(continuous, resumed), "resumed codes must equal the uninterrupted run")
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path), "a completed run removes its checkpoint")
    }

    func testNaNAtEpochZeroThrowsDiverged() throws {
        XCTAssertThrowsError(try run(config(seed: 7)) { $0.lossOverride = { _, _ in .nan } }) { error in
            XCTAssertTrue(error is VoxtralVoiceEnrollment.EnrollmentDivergedError, "\(error)")
        }
    }

    func testNaNAtEpochKReturnsBestStepCodes() throws {
        let k = 5
        nonisolated(unsafe) var paramsBeforeK: (MLXArray, MLXArray)?
        var captured: VoxtralVoiceEnrollment?
        let codes = try run(config(seed: 7)) { enroller in
            captured = enroller
            // Decreasing losses up to k - 1 make epoch k - 1 the best step; NaN at k
            enroller.lossOverride = { epoch, _ in epoch < k ? Float(100 - epoch) : .nan }
            enroller.epochObserver = { epoch, s, a in if epoch == k - 1 { paramsBeforeK = (s, a) } }
        }
        let (s, a) = try XCTUnwrap(paramsBeforeK)
        let best = try XCTUnwrap(captured).discreteCodes(semanticLogits: s, acousticValues: a)
        print("[enroll] NaN at epoch \(k): codes == best step (epoch \(k - 1)) → \(identical(codes, best))")
        XCTAssertTrue(identical(codes, best))
    }

    func testCancellationThrowsCancellationError() throws {
        XCTAssertThrowsError(try run(config(seed: 7), stopAt: 3)) { error in
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
    }
}
