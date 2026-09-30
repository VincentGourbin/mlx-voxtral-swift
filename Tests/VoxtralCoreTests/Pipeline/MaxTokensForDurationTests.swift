/**
 * MaxTokensForDurationTests - K-5 (P-11, ASK-8 = A)
 *
 * STT `maxTokens` defaults to nil: the budget follows the audio duration (⌈s × 6⌉ + 64, at least
 * 500), so 11 minutes are no longer cut at 500 tokens; an explicit value is a cap, and reaching it
 * is reported by `lastResultTruncated`.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class MaxTokensForDurationTests: XCTestCase {

    func testDefaultIsAutomatic() {
        XCTAssertNil(VoxtralPipeline.Configuration.default.maxTokens)
        XCTAssertNil(VoxtralPipeline.Configuration().maxTokens)
    }

    func testBudgetFollowsDuration() {
        XCTAssertEqual(VoxtralPipeline.automaticMaxTokens(forDuration: 5), 500)          // C-court: former floor
        XCTAssertEqual(VoxtralPipeline.automaticMaxTokens(forDuration: 167), 1_066)      // C-moyen EN (520 needed)
        XCTAssertEqual(VoxtralPipeline.automaticMaxTokens(forDuration: 681.44), 4_153)   // C-long (≈ 2 430 needed)
        // Stays ≥ 1.5 × the densest measured speech rate (FR ≈ 4.0 tokens/s)
        XCTAssertGreaterThanOrEqual(VoxtralPipeline.automaticTokensPerSecond, 1.5 * 693 / 173.8)
    }

    /// Heavy (mini-3b-8bit): an explicit cap below the need is reported; the default is not truncated.
    func testTruncationIsReported() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_LONG_AUDIO"] == "1",
                          "Set VOXTRAL_LONG_AUDIO=1 to run")
        let audio = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("docs/examples/fluxforge_short_en_6bit.wav")
        let capped = VoxtralPipeline(model: .mini3b8bit, backend: .mlx, configuration: .init(maxTokens: 5))
        try await capped.loadModel()
        _ = try await capped.transcribe(audio: audio, language: "en")
        XCTAssertTrue(capped.lastResultTruncated, "a 5-token cap must be reported")
        capped.unload()

        let automatic = VoxtralPipeline(model: .mini3b8bit, backend: .mlx)
        try await automatic.loadModel()
        _ = try await automatic.transcribe(audio: audio, language: "en")
        XCTAssertFalse(automatic.lastResultTruncated)
        automatic.unload()
    }
}
