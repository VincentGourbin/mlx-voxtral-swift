/**
 * TTSFrameCapTests - K-14
 *
 * The TTS frame cap follows the text: min(maxFrames, 70 + ⌈10.4 × text tokens⌉) by default, so a missed end of audio
 * costs about 3 × the expected duration instead of 2 500 frames (200 s of babble, up to 16 min of compute in bf16).
 */

import XCTest
@testable import VoxtralCore

final class TTSFrameCapTests: XCTestCase {

    func testDefaultCapFollowsTheText() {
        let configuration = VoxtralTTSPipeline.Configuration.default
        XCTAssertEqual(VoxtralTTSPipeline.frameCap(textTokens: 7, configuration: configuration), 143)  // "Bonjour, comment ça va ?"
        XCTAssertEqual(VoxtralTTSPipeline.frameCap(textTokens: 49, configuration: configuration), 580)
        XCTAssertEqual(VoxtralTTSPipeline.frameCap(textTokens: 333, configuration: configuration), 2500)  // bounded by maxFrames
    }

    func testCustomConfigurationsGetTheCapToo() {
        let configuration = VoxtralTTSPipeline.Configuration(maxFrames: 1000)
        XCTAssertEqual(VoxtralTTSPipeline.frameCap(textTokens: 10, configuration: configuration), 174)
    }

    func testNilKeepsTheFixedMaxFrames() {
        let configuration = VoxtralTTSPipeline.Configuration(maxFrames: 2500, framesPerTextToken: nil)
        XCTAssertEqual(VoxtralTTSPipeline.frameCap(textTokens: 7, configuration: configuration), 2500)
    }
}
