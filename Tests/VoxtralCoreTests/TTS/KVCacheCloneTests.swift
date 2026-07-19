/**
 * KVCacheCloneTests — the voice-prefix KV cache invariant (Fluxforge ask A6).
 *
 * `cloneKVCaches` must produce clones whose subsequent autoregressive
 * updates NEVER leak back into the source caches: the pipeline memoizes the
 * voice prefix KV per voice and reuses it across syntheses, so any aliasing
 * would corrupt every synthesis after the first.
 *
 * The safety rests on MLXLMCommon's `KVCacheSimple` internals (state getter
 * slicing to `offset`, update() reallocating when capacity is exhausted) —
 * these tests pin that contract, including the exact-capacity edge case
 * where `state` returns the source's own MLXArray instances.
 */

import XCTest
import MLX
import MLXLMCommon
import MLXRandom
@testable import VoxtralCore

final class KVCacheCloneTests: XCTestCase {

    private func snapshot(_ c: any KVCache) -> (offset: Int, sums: [Float]) {
        (c.offset, c.state.map { $0.asType(.float32).sum().item(Float.self) })
    }

    /// Fill a fresh KVCacheSimple with `t` random positions.
    private func makeCache(t: Int, heads: Int = 2, dim: Int = 4) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let k = MLXRandom.normal([1, heads, t, dim])
        let v = MLXRandom.normal([1, heads, t, dim])
        _ = cache.update(keys: k, values: v)
        MLX.eval(cache.state)
        return cache
    }

    private func assertCloneIsolation(prefixLen: Int, file: StaticString = #filePath, line: UInt = #line) {
        let source = makeCache(t: prefixLen)
        let before = snapshot(source)

        // Two consecutive "syntheses", each cloning then extending the clone
        // (mirrors generate()'s suffix prefill + autoregressive decode).
        for _ in 0..<2 {
            let clone = cloneKVCaches([source])[0]
            XCTAssertEqual(clone.offset, prefixLen, file: file, line: line)
            for _ in 0..<3 {
                let k = MLXRandom.normal([1, 2, 5, 4])
                _ = clone.update(keys: k, values: k)
            }
            MLX.eval(clone.state)
            XCTAssertEqual(clone.offset, prefixLen + 15, file: file, line: line)
        }

        let after = snapshot(source)
        XCTAssertEqual(before.offset, after.offset,
                       "clone updates changed the source cache offset", file: file, line: line)
        XCTAssertEqual(before.sums, after.sums,
                       "clone updates leaked into the source cache contents", file: file, line: line)
    }

    /// Typical case: offset < buffer capacity → state returns trimmed views.
    func testCloneUpdatesDoNotMutateSource() {
        MLXRandom.seed(7)
        assertCloneIsolation(prefixLen: 103)
    }

    /// Edge case: offset == buffer capacity (prefix length exactly a multiple
    /// of KVCacheSimple's 256 step) → state returns the source's own MLXArray
    /// instances, shared with the clone until its first update() rebinds.
    func testCloneUpdatesDoNotMutateSourceAtExactCapacity() {
        MLXRandom.seed(7)
        assertCloneIsolation(prefixLen: 256)
    }

    /// The clone must equal the source at the moment of cloning (same KV
    /// content and offset) — a deep copy, not a reset cache.
    func testCloneMatchesSourceState() {
        MLXRandom.seed(7)
        let source = makeCache(t: 64)
        let clone = cloneKVCaches([source])[0]
        XCTAssertEqual(clone.offset, source.offset)
        XCTAssertEqual(snapshot(clone).sums, snapshot(source).sums)
    }
}
