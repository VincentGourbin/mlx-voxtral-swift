/**
 * PadFractionTests — K-89: `pad_fraction` and `streaming_pad_fraction` of `bench realtime` on a fixed token sequence
 * (`VoxtralRealtimePipeline.padFractions`, the values behind `lastPadFraction` / `lastStreamingPadFraction`).
 */

import XCTest
@testable import VoxtralCore

final class PadFractionTests: XCTestCase {

    func testPadFractionsOfAFixedSequence() {
        // 32 = [STREAMING_PAD], 33 = [STREAMING_WORD] (both carry no text), 5 and 7 = text tokens
        let tokens: [Int] = [32, 32, 5, 33, 7, 32, 32, 33]
        let (pad, streamingPad) = VoxtralRealtimePipeline.padFractions(tokens, silent: [32, 33], streamingPad: 32)
        XCTAssertEqual(pad, 6.0 / 8.0)
        XCTAssertEqual(streamingPad, 4.0 / 8.0)
    }

    func testPadFractionsOfAnEmptySequenceAreNil() {
        let (pad, streamingPad) = VoxtralRealtimePipeline.padFractions([], silent: [32], streamingPad: 32)
        XCTAssertNil(pad)
        XCTAssertNil(streamingPad)
    }
}
