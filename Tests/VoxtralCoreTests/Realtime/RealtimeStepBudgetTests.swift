/**
 * RealtimeStepBudgetTests - K-5 (P-64)
 *
 * The Realtime decode loop takes one step per audio frame: `maxTokens` is a budget of *text*
 * tokens (streaming pads and specials are free), reached with `>=`, and reported as truncated.
 * Before K-5 it counted every step, so audio longer than 4 096 frames (≈ 5 min 27 s) was cut
 * silently at step 4 097.
 */

import XCTest
@testable import VoxtralCore

final class RealtimeStepBudgetTests: XCTestCase {

    private let pad = 11, eos = 2, text = 1_500

    func testStepsEqualFrames() {
        let frames = 5_000
        let loop = VoxtralRealtimeModel.decodeLoop(
            promptLen: 0, nAudioTotal: frames, maxTextTokens: 4_096, eosTokenId: eos,
            sample: { self.pad }, advance: { _, _, _ in true })
        XCTAssertEqual(loop.steps, frames, "steps=\(loop.steps) frames=\(frames)")
        XCTAssertFalse(loop.truncated)
    }

    func testTextBudgetIsReachedExactlyAndReported() {
        let loop = VoxtralRealtimeModel.decodeLoop(
            promptLen: 0, nAudioTotal: 5_000, maxTextTokens: 100, eosTokenId: eos,
            sample: { self.text }, advance: { _, _, _ in true })
        XCTAssertEqual(loop.tokens.count, 100, "the budget is reached with >=, not one past it")
        XCTAssertTrue(loop.truncated)
    }
}
