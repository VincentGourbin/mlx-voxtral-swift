/**
 * MLXCachePolicyTests - K-52
 *
 * The MLX buffer-cache limit is opt-in and process-wide: a pipeline sets it only when asked and
 * gives the host's value back when it unloads.
 */

import MLX
import XCTest
@testable import VoxtralCore

final class MLXCachePolicyTests: XCTestCase {

    func testNilLeavesHostLimitAlone() {
        let host = Memory.cacheLimit
        let policy = MLXCachePolicy()
        policy.apply(nil)
        XCTAssertFalse(policy.isActive)
        XCTAssertEqual(Memory.cacheLimit, host)
        policy.restore()
        XCTAssertEqual(Memory.cacheLimit, host)
    }

    func testLimitIsAppliedThenHostValueRestored() {
        let host = Memory.cacheLimit
        defer { Memory.cacheLimit = host }
        let policy = MLXCachePolicy()
        policy.apply(256 * 1_048_576)
        XCTAssertTrue(policy.isActive)
        XCTAssertEqual(Memory.cacheLimit, 256 * 1_048_576)
        policy.apply(512 * 1_048_576)  // a reload keeps the host value, not the previous limit
        policy.restore()
        XCTAssertFalse(policy.isActive)
        XCTAssertEqual(Memory.cacheLimit, host)
    }

    func testPresetsDoNotSetACacheLimit() {
        for config in [MemoryOptimizationConfig.light, .moderate, .aggressive, .ultra, .disabled, .recommended()] {
            XCTAssertNil(config.cacheLimitBytes)
        }
        XCTAssertNil(VoxtralTTSPipeline.Configuration.default.cacheLimitBytes)
        XCTAssertNil(VoxtralRealtimePipeline.Configuration.default.cacheLimitBytes)
    }
}
