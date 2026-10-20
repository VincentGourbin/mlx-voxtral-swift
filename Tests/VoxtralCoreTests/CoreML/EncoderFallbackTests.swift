/**
 * EncoderFallbackTests - K-87
 *
 * `.auto` / `.hybrid`: the Core ML encoder is used when it can be had (from the cache: no download, so it works
 * offline); when it cannot (absent and no network), the MLX encoder is used — the public behaviour, no error — and the
 * fallback is announced at `notice` level in the unified log even with `VoxtralDebug.enabled = false`.
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class EncoderFallbackTests: XCTestCase {

    private struct Offline: LocalizedError { var errorDescription: String? { "The Internet connection appears to be offline." } }

    override func tearDown() {
        VoxtralDebug.noticeObserver = nil
        super.tearDown()
    }

    func testCachedCoreMLEncoderIsUsedWithoutNotice() async {
        var notices: [String] = []
        VoxtralDebug.noticeObserver = { notices.append($0) }
        let chosen = await VoxtralPipeline.encoder(download: { "coreml" }, fallback: { "mlx" })
        XCTAssertEqual(chosen, "coreml")
        XCTAssertEqual(notices, [])
    }

    func testMissingCoreMLEncoderFallsBackToMLXAndSaysSo() async {
        let saved = VoxtralDebug.enabled
        VoxtralDebug.enabled = false
        defer { VoxtralDebug.enabled = saved }
        var notices: [String] = []
        VoxtralDebug.noticeObserver = { notices.append($0) }
        let chosen = await VoxtralPipeline.encoder(download: { () async throws -> String in throw Offline() },
                                                   fallback: { "mlx" })
        XCTAssertEqual(chosen, "mlx", "the fallback keeps .auto working (public behaviour)")
        XCTAssertEqual(notices.count, 1)
        XCTAssertTrue(notices.first?.contains("using the MLX encoder") == true, "\(notices)")
        XCTAssertTrue(notices.first?.contains("offline") == true, "the notice gives the cause: \(notices)")
    }
}
