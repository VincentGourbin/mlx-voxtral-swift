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
/// fingerprint, never by key: the key only labels an entry, a key reused with another embedding recomputes, and
/// voices sharing a key (every `voiceEmbedding:` call uses "embedding") keep their own entries.
/// Least recently used entry evicted beyond `capacity` (≈ 27 MB per entry in tts-4b, K-50). Entries are never mutated:
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

    /// Cached prefix of `embedding`, else `compute`'s; not kept when `failed()` is true after computing (an MLX error
    /// caught in the enclosing boundary leaves empty arrays).
    func prefix(
        key: String, embedding: MLXArray, failed: () -> Bool = { false },
        compute: (MLXArray) -> (cache: [any KVCache], prefixLen: Int)
    ) -> (cache: [any KVCache], len: Int) {
        let fingerprint = VoiceFingerprint(embedding)
        if let index = entries.firstIndex(where: { $0.fingerprint == fingerprint }) {
            let entry = entries.remove(at: index)
            entries.append(entry)
            return (entry.cache, entry.len)
        }
        let (cache, len) = compute(embedding)
        if failed() { return (cache, len) }
        entries.append(Entry(key: key, fingerprint: fingerprint, cache: cache, len: len))
        if entries.count > capacity { entries.removeFirst(entries.count - capacity) }
        return (cache, len)
    }

    func removeAll() {
        entries.removeAll()
    }
}
