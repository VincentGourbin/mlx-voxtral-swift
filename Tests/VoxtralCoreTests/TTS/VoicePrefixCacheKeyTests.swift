/**
 * VoicePrefixCacheKeyTests — K-50 (P-40): the TTS voice-prefix cache is matched by embedding content, not by key.
 *
 * Before K-50 the pipeline kept one entry keyed by a string (`voice.rawValue` or the streaming `voiceKey`): the same
 * key passed with another embedding reused the stale prefix, and the batch `voiceEmbedding:` overload had no cache.
 * Fake prefix computation (no model): counts how many prefixes are computed and checks which one is returned.
 */

import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import XCTest
@testable import VoxtralCore

final class VoicePrefixCacheKeyTests: XCTestCase {

    private var computed = 0

    /// Fake prefix: a one-layer cache whose offset is the embedding's frame count; `len` tags the call
    private func compute(_ embedding: MLXArray) -> (cache: [any KVCache], prefixLen: Int) {
        computed += 1
        let cache = KVCacheSimple()
        let kv = MLXArray.zeros([1, 1, embedding.dim(0), 2])
        _ = cache.update(keys: kv, values: kv)
        return ([cache], 1000 * computed + embedding.dim(0))
    }

    private func voice(_ seed: UInt64, frames: Int = 5) -> MLXArray {
        MLXRandom.normal([frames, 8], key: MLXRandom.key(seed)).asType(.float16)
    }

    func testKeyReusedWithAnotherEmbeddingIsDetected() {
        let cache = VoicePrefixCache()
        let first = cache.prefix(key: "clone", embedding: voice(1), compute: compute)
        let second = cache.prefix(key: "clone", embedding: voice(2), compute: compute)
        XCTAssertEqual(computed, 2, "same key, other embedding: the prefix must be recomputed")
        XCTAssertNotEqual(first.len, second.len)
        print("[k50] key reused with another embedding: \(computed) prefixes computed (stale entry not reused)")
    }

    /// Every `voiceEmbedding:` call labels its entry "embedding": two voices used in turn keep their own entries.
    func testVoicesSharingAKeyKeepTheirEntries() {
        let cache = VoicePrefixCache()
        let a = cache.prefix(key: "embedding", embedding: voice(1), compute: compute)
        let b = cache.prefix(key: "embedding", embedding: voice(2), compute: compute)
        let a2 = cache.prefix(key: "embedding", embedding: voice(1), compute: compute)
        let b2 = cache.prefix(key: "embedding", embedding: voice(2), compute: compute)
        XCTAssertEqual(computed, 2, "A, B, A, B: each voice's prefix computed once")
        XCTAssertEqual(a.len, a2.len)
        XCTAssertEqual(b.len, b2.len)
        XCTAssertNotEqual(a.len, b.len)
        XCTAssertEqual(cache.entries.count, 2)
    }

    func testSameEmbeddingIsReusedWhateverTheKey() {
        let cache = VoicePrefixCache()
        let first = cache.prefix(key: "embedding", embedding: voice(1), compute: compute)
        // Same content in a fresh array (a cloned voice loaded twice), under another label
        let again = cache.prefix(key: "fr_female", embedding: voice(1), compute: compute)
        XCTAssertEqual(computed, 1)
        XCTAssertEqual(first.len, again.len)
        XCTAssertTrue((first.cache[0] as AnyObject) === (again.cache[0] as AnyObject))
    }

    func testFingerprintSeesShapeDTypeAndContent() {
        let a = voice(1)
        XCTAssertEqual(VoiceFingerprint(a), VoiceFingerprint(voice(1)))
        XCTAssertNotEqual(VoiceFingerprint(a), VoiceFingerprint(voice(2)))
        XCTAssertNotEqual(VoiceFingerprint(a), VoiceFingerprint(a.asType(.float32)))
        XCTAssertNotEqual(VoiceFingerprint(a), VoiceFingerprint(a.reshaped(8, 5)))
        // One element changed
        var values = a.asType(.float32).asArray(Float.self)
        values[17] += 1
        XCTAssertNotEqual(VoiceFingerprint(a), VoiceFingerprint(MLXArray(values, a.shape).asType(.float16)))
    }

    /// A prefix computed while an MLX error was caught is not kept: the next call recomputes
    func testFailedPrefixIsNotKept() {
        let cache = VoicePrefixCache()
        _ = cache.prefix(key: "embedding", embedding: voice(1), failed: { true }, compute: compute)
        XCTAssertTrue(cache.entries.isEmpty)
        _ = cache.prefix(key: "embedding", embedding: voice(1), compute: compute)
        XCTAssertEqual(computed, 2)
        XCTAssertEqual(cache.entries.count, 1)
    }

    func testLeastRecentlyUsedEviction() {
        let cache = VoicePrefixCache(capacity: 3)
        for seed in UInt64(1) ... 3 { _ = cache.prefix(key: "v\(seed)", embedding: voice(seed), compute: compute) }
        _ = cache.prefix(key: "v1", embedding: voice(1), compute: compute)  // v1 becomes the most recent
        XCTAssertEqual(computed, 3)
        _ = cache.prefix(key: "v4", embedding: voice(4), compute: compute)  // evicts v2
        XCTAssertEqual(cache.entries.map(\.key), ["v3", "v1", "v4"])
        _ = cache.prefix(key: "v1", embedding: voice(1), compute: compute)
        XCTAssertEqual(computed, 4, "v1 kept")
        _ = cache.prefix(key: "v2", embedding: voice(2), compute: compute)
        XCTAssertEqual(computed, 5, "v2 was evicted")
        XCTAssertEqual(cache.entries.count, 3)
        cache.removeAll()
        XCTAssertTrue(cache.entries.isEmpty)
    }
}
