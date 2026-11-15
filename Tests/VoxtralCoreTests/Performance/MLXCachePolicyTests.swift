/**
 * MLXCachePolicyTests - K-52
 *
 * The MLX limits are process-wide: a pipeline sets only what its configuration asks and gives the
 * host's values back when it unloads.
 */

import MLX
import XCTest
@testable import VoxtralCore

final class MLXCachePolicyTests: XCTestCase {

    func testNoLimitLeavesTheHostSettingsAlone() {
        let (cache, memory) = (Memory.cacheLimit, Memory.memoryLimit)
        let policy = MLXCachePolicy()
        policy.apply(.none)
        XCTAssertFalse(policy.isActive)
        XCTAssertEqual(Memory.cacheLimit, cache)
        policy.enter(.generate)
        XCTAssertEqual(Memory.cacheLimit, cache)
        policy.restore()
        XCTAssertEqual(Memory.cacheLimit, cache)
        XCTAssertEqual(Memory.memoryLimit, memory)
    }

    func testLimitIsAppliedThenHostValueRestored() {
        let host = Memory.cacheLimit
        defer { Memory.cacheLimit = host }
        let policy = MLXCachePolicy()
        policy.apply(VoxtralMemoryLimits(cacheBytes: 256 * 1_048_576))
        XCTAssertTrue(policy.isActive)
        XCTAssertEqual(Memory.cacheLimit, 256 * 1_048_576)
        policy.apply(VoxtralMemoryLimits(cacheBytes: 512 * 1_048_576))  // a reload keeps the host value, not the previous limit
        policy.restore()
        XCTAssertFalse(policy.isActive)
        XCTAssertEqual(Memory.cacheLimit, host)
    }

    func testEachStageHasItsCeiling() {
        let host = Memory.cacheLimit
        defer { Memory.cacheLimit = host }
        let policy = MLXCachePolicy()
        policy.apply(VoxtralMemoryLimits(cacheBytes: 2048 * 1_048_576))
        XCTAssertEqual(Memory.cacheLimit, 1024 * 1_048_576, "loading")
        policy.enter(.generate)
        XCTAssertEqual(Memory.cacheLimit, 2048 * 1_048_576)
        policy.restore()
    }

    func testTheTightestPipelineWinsAndTheHostComesBackWithTheLast() {
        let host = Memory.cacheLimit
        defer { Memory.cacheLimit = host }
        let stt = MLXCachePolicy()
        let tts = MLXCachePolicy()
        stt.apply(VoxtralMemoryLimits(cacheBytes: 512 * 1_048_576))
        tts.apply(VoxtralMemoryLimits(cacheBytes: 256 * 1_048_576))
        XCTAssertEqual(Memory.cacheLimit, 256 * 1_048_576)
        stt.enter(.generate)
        XCTAssertEqual(Memory.cacheLimit, 256 * 1_048_576, "a stage of one pipeline does not lift the other's ceiling")

        tts.restore()
        XCTAssertEqual(Memory.cacheLimit, 512 * 1_048_576, "the first pipeline keeps its ceiling")
        stt.restore()
        XCTAssertEqual(Memory.cacheLimit, host)
    }

    func testAPipelineDroppedWithoutUnloadGivesItsLimitsBack() {
        let host = Memory.cacheLimit
        defer { Memory.cacheLimit = host }
        var policy: MLXCachePolicy? = MLXCachePolicy()
        policy?.apply(VoxtralMemoryLimits(cacheBytes: 256 * 1_048_576))
        XCTAssertEqual(Memory.cacheLimit, 256 * 1_048_576)
        policy = nil
        XCTAssertEqual(Memory.cacheLimit, host)
    }

    func testMemoryLimitIsAppliedThenHostValueRestored() {
        let host = Memory.memoryLimit
        defer { Memory.memoryLimit = host }
        let policy = MLXCachePolicy()
        policy.apply(VoxtralMemoryLimits(memoryBytes: 4096 * 1_048_576))
        XCTAssertEqual(Memory.memoryLimit, 4096 * 1_048_576)
        policy.restore()
        XCTAssertEqual(Memory.memoryLimit, host)
    }

    func testLeanEmptiesTheCacheBetweenStages() {
        let host = Memory.cacheLimit
        defer { Memory.cacheLimit = host }
        func fillCache() {
            var buffer: MLXArray? = MLXArray.ones([1_024, 1_024]) + 1
            MLX.eval(buffer!)
            buffer = nil
            Stream.gpu.synchronize()
        }
        let policy = MLXCachePolicy()

        policy.apply(VoxtralMemoryLimits(cacheBytes: 512 * 1_048_576))
        fillCache()
        XCTAssertGreaterThan(Memory.cacheMemory, 0)
        policy.enter(.generate)
        XCTAssertGreaterThan(Memory.cacheMemory, 0, "a stage alone keeps the buffers")
        policy.restore()

        policy.apply(VoxtralMemoryLimits(cacheBytes: 512 * 1_048_576, clearsCacheBetweenStages: true))
        fillCache()
        XCTAssertGreaterThan(Memory.cacheMemory, 0)
        policy.enter(.generate)
        XCTAssertEqual(Memory.cacheMemory, 0)
        policy.restore()
    }
}
