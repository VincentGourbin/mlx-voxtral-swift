/**
 * MLXCachePolicy - the MLX limits one pipeline holds while it is loaded (K-52, P-09, P-42, P-67)
 *
 * Without `Memory.cacheLimit`, MLX keeps every freed buffer for reuse: on C-long (11 min) the STT process reached 70 GB
 * for 8.6 GB of active MLX memory, and the machine swapped. The limits are process-wide, so a pipeline sets only what
 * its configuration asks, from the start of loading, and gives the previous values back when it unloads.
 */

import MLX

final class MLXCachePolicy: Sendable {
    private struct Slot {
        var limits = VoxtralMemoryLimits.none
        /// Values before this pipeline set its own, restored by `restore()`
        var savedCacheLimit: Int?
        var savedMemoryLimit: Int?
    }

    private let slot = Locked(Slot())

    /// Sets `limits` for loading, remembering the host's values once; nil fields leave them alone.
    func apply(_ limits: VoxtralMemoryLimits) {
        slot.withLock { slot in
            if limits.cacheBytes != nil, slot.savedCacheLimit == nil { slot.savedCacheLimit = Memory.cacheLimit }
            if limits.memoryBytes != nil, slot.savedMemoryLimit == nil { slot.savedMemoryLimit = Memory.memoryLimit }
            slot.limits = limits
        }
        if let bytes = limits.memoryBytes { Memory.memoryLimit = bytes }
        if let bytes = limits.cacheBytes(for: .load) { Memory.cacheLimit = bytes }
    }

    /// A stage starts: its cache ceiling applies and, if asked, what the previous stage left in the cache goes.
    func enter(_ stage: VoxtralMemoryLimits.Stage) {
        let limits = slot.withLock { $0.limits }
        if let bytes = limits.cacheBytes(for: stage) { Memory.cacheLimit = bytes }
        if limits.clearsCacheBetweenStages { Memory.clearCache() }
    }

    /// True while this pipeline holds a cache limit
    var isActive: Bool { slot.withLock { $0.savedCacheLimit != nil } }

    /// Frees the cached buffers after a response when a limit is set.
    func endOfResponse() {
        if isActive { Memory.clearCache() }
    }

    /// Restores the host's limits (if this pipeline changed them) and frees the cache.
    func restore() {
        let saved = slot.withLock { slot -> Slot in
            let saved = slot
            slot = Slot()
            return saved
        }
        if let cache = saved.savedCacheLimit { Memory.cacheLimit = cache }
        if let memory = saved.savedMemoryLimit { Memory.memoryLimit = memory }
        Memory.clearCache()
    }
}
