import CryptoKit
import Foundation
import MLX
import MLXLMCommon

/// Content fingerprint of a voice embedding: shape, dtype and SHA-256 of its bytes (K-50).
struct VoiceFingerprint: Hashable {
    let shape: [Int]
    let dtype: String
    let digest: String

    init(_ embedding: MLXArray) {
        let data = embedding.asData(access: .copy)
        shape = data.shape
        dtype = String(describing: data.dType)
        digest = SHA256.hash(data: data.data).map { String(format: "%02x", $0) }.joined()
    }
}

/// Voice-prefix KV caches of the last voices used, matched by embedding content (K-50).
///
/// The voice frames precede the text in the prompt, so their KV depends only on the voice embedding: any voice
/// (preset, cloned, ZeroVoice, blend) reuses its prefix when the same embedding comes back. Entries are matched by
/// fingerprint, never by key alone: a key reused with another embedding drops its stale entry and recomputes.
/// Least recently used entry evicted beyond `capacity` (≈ 23 MB per entry in tts-4b). Entries are never mutated:
/// generation clones them (`cloneKVCaches`, KVCacheCloneTests).
final class VoicePrefixCache {
    struct Entry {
        let key: String
        let fingerprint: VoiceFingerprint
        let cache: [any KVCache]
        let len: Int
    }

    let capacity: Int
    /// Least recently used first
    private(set) var entries: [Entry] = []

    init(capacity: Int = 3) {
        self.capacity = capacity
    }

    func prefix(
        key: String, embedding: MLXArray, compute: (MLXArray) -> (cache: [any KVCache], prefixLen: Int)
    ) -> (cache: [any KVCache], len: Int) {
        let fingerprint = VoiceFingerprint(embedding)
        if let index = entries.firstIndex(where: { $0.fingerprint == fingerprint }) {
            let entry = entries.remove(at: index)
            entries.append(entry)
            return (entry.cache, entry.len)
        }
        // A key reused with another embedding: its stale entry goes
        entries.removeAll { $0.key == key }
        let (cache, len) = compute(embedding)
        entries.append(Entry(key: key, fingerprint: fingerprint, cache: cache, len: len))
        if entries.count > capacity { entries.removeFirst(entries.count - capacity) }
        return (cache, len)
    }

    func removeAll() {
        entries.removeAll()
    }
}
