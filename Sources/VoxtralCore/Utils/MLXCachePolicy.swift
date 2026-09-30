/**
 * MLXCachePolicy - opt-in bound on MLX's buffer cache for one pipeline (K-52, P-09, P-42, P-67)
 *
 * Without `Memory.cacheLimit`, MLX keeps every freed buffer for reuse: on C-long (11 min) the STT
 * process reached 70 GB for 8.6 GB of active MLX memory, and the machine swapped. The limit is
 * process-wide, so it is opt-in (`nil` leaves the host's setting alone) and the previous value is
 * restored when the pipeline unloads.
 */

import MLX

final class MLXCachePolicy: Sendable {
    /// `Memory.cacheLimit` before this pipeline set its own, restored by `restore()`
    private let previous = Locked<Int?>(nil)

    /// Sets the cache limit to `bytes` (no-op when nil), remembering the host's value once.
    func apply(_ bytes: Int?) {
        guard let bytes else { return }
        previous.withLock { if $0 == nil { $0 = Memory.cacheLimit } }
        Memory.cacheLimit = bytes
    }

    /// True while this pipeline holds a limit
    var isActive: Bool { previous.get() != nil }

    /// Frees the cached buffers after a response when a limit is set.
    func endOfResponse() {
        if isActive { Memory.clearCache() }
    }

    /// Restores the host's limit (if this pipeline changed it) and frees the cache.
    func restore() {
        let saved = previous.withLock { value -> Int? in
            let saved = value
            value = nil
            return saved
        }
        if let saved { Memory.cacheLimit = saved }
        Memory.clearCache()
    }
}
