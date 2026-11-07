/**
 * MLXCachePolicy - the MLX limits one pipeline holds while it is loaded (K-52, P-09, P-42, P-67)
 *
 * Without `Memory.cacheLimit`, MLX keeps every freed buffer for reuse: on C-long (11 min) the STT process reached 70 GB
 * for 8.6 GB of active MLX memory, and the machine swapped. The limits are process-wide: with several pipelines loaded,
 * the tightest applies, and the host's values come back with the last one.
 */

import MLX

final class MLXCachePolicy: Sendable {
    private struct ProcessLimits {
        var held: [ObjectIdentifier: (cache: Int?, memory: Int?)] = [:]
        /// Values before the first pipeline set its own
        var hostCacheLimit: Int?
        var hostMemoryLimit: Int?

        mutating func update() {
            if let cache = held.values.compactMap(\.cache).min() {
                if hostCacheLimit == nil { hostCacheLimit = Memory.cacheLimit }
                Memory.cacheLimit = cache
            } else if let host = hostCacheLimit {
                Memory.cacheLimit = host
                hostCacheLimit = nil
            }
            if let memory = held.values.compactMap(\.memory).min() {
                if hostMemoryLimit == nil { hostMemoryLimit = Memory.memoryLimit }
                Memory.memoryLimit = memory
            } else if let host = hostMemoryLimit {
                Memory.memoryLimit = host
                hostMemoryLimit = nil
            }
        }
    }

    private static let process = Locked(ProcessLimits())

    private let limits = Locked(VoxtralMemoryLimits.none)

    deinit { restore() }

    /// Sets `limits`, for loading; nil fields leave the host's settings alone.
    func apply(_ limits: VoxtralMemoryLimits) {
        self.limits.set(limits)
        hold(.load)
    }

    /// A stage starts: its cache ceiling applies and, if asked, what the previous stage left in the cache goes.
    func enter(_ stage: VoxtralMemoryLimits.Stage) {
        hold(stage)
        if limits.get().clearsCacheBetweenStages { Memory.clearCache() }
    }

    private func hold(_ stage: VoxtralMemoryLimits.Stage) {
        let limits = limits.get()
        Self.process.withLock { process in
            process.held[ObjectIdentifier(self)] = (limits.cacheBytes(for: stage), limits.memoryBytes)
            process.update()
        }
    }

    /// True while this pipeline holds a cache limit
    var isActive: Bool { Self.process.withLock { $0.held[ObjectIdentifier(self)]?.cache != nil } }

    /// Frees the cached buffers after a response when a limit is set.
    func endOfResponse() {
        if isActive { Memory.clearCache() }
    }

    /// Gives up this pipeline's limits and frees the cache.
    func restore() {
        Self.process.withLock { process in
            process.held[ObjectIdentifier(self)] = nil
            process.update()
        }
        limits.set(.none)
        Memory.clearCache()
    }
}
