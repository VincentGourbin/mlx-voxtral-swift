/**
 * DeprecatedNames - the 2.x names of the renamed public types (K-31, ASK-25): generic names collided with the
 * hosts' own types (FluxForge `ModelRegistry`, `ModelDownloader`; LTX `RuntimeBeacon`). Kept through 3.x.
 */

import Foundation

@available(*, deprecated, renamed: "VoxtralModelRegistry")
public typealias ModelRegistry = VoxtralModelRegistry

@available(*, deprecated, renamed: "VoxtralModelDownloader")
public typealias ModelDownloader = VoxtralModelDownloader

@available(*, deprecated, renamed: "VoxtralModelDownloaderError")
public typealias ModelDownloaderError = VoxtralModelDownloaderError

@available(*, deprecated, renamed: "VoxtralDownloadProgressCallback")
public typealias DownloadProgressCallback = VoxtralDownloadProgressCallback

@available(*, deprecated, renamed: "VoxtralRuntimeBeacon")
public typealias RuntimeBeacon = VoxtralRuntimeBeacon

// MARK: - Memory presets: `.fast`, `.lean` and `VoxtralMemoryLimits` replace them

extension MemoryOptimizationConfig {
    @available(*, deprecated, message: "No flush runs inside the decoding loop any more (K-51): use .fast or .lean")
    public init(
        evalFrequency: Int, clearCacheOnEval: Bool = false, resetPeakMemory: Bool = false, maxKVCacheSize: Int? = nil,
        cacheLimitBytes: Int? = nil
    ) {
        self.init(limits: VoxtralMemoryLimits(cacheBytes: cacheLimitBytes), maxKVCacheSize: maxKVCacheSize)
    }

    @available(*, deprecated, message: "No flush runs inside the decoding loop any more (K-51): use .fast or .lean")
    public init(clearCacheOnEval: Bool, resetPeakMemory: Bool = false, maxKVCacheSize: Int? = nil) {
        self.init(maxKVCacheSize: maxKVCacheSize)
    }

    @available(*, deprecated, message: "No effect since K-32: a measuring tool resets the peak counter")
    public init(resetPeakMemory: Bool, maxKVCacheSize: Int? = nil) {
        self.init(maxKVCacheSize: maxKVCacheSize)
    }

    @available(*, deprecated, renamed: "fast")
    public static var disabled: MemoryOptimizationConfig { .fast }
    @available(*, deprecated, renamed: "fast")
    public static var light: MemoryOptimizationConfig { .fast }
    @available(*, deprecated, renamed: "fast")
    public static var moderate: MemoryOptimizationConfig { .fast }
    @available(*, deprecated, message: "Use .recommended(forRAMGB:) or .lean")
    public static var aggressive: MemoryOptimizationConfig { .recommended(forRAMGB: 16) }
    @available(*, deprecated, message: "Use .recommended(forRAMGB:) or .lean")
    public static var ultra: MemoryOptimizationConfig { .recommended(forRAMGB: 8) }

    @available(*, deprecated, message: "No effect: no flush runs inside the decoding loop any more (K-51)")
    public var evalFrequency: Int { get { 0 } set { _ = newValue } }
    @available(*, deprecated, message: "No effect: no flush runs inside the decoding loop any more (K-51)")
    public var clearCacheOnEval: Bool { get { false } set { _ = newValue } }
    @available(*, deprecated, message: "No effect since K-32: a measuring tool resets the peak counter")
    public var resetPeakMemory: Bool { get { false } set { _ = newValue } }
}
