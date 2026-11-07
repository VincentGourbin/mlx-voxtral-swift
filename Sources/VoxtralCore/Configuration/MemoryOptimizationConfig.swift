/**
 * MemoryOptimizationConfig - memory settings of a transcription or chat pipeline
 *
 * `.fast` keeps everything resident; `.lean` trades a little time for the lowest peak. Each field is a setting of its
 * own, so that a reference profile can fix them one by one.
 */

import Foundation

public struct MemoryOptimizationConfig: Sendable, Equatable, CustomStringConvertible {

    /// MLX limits held while the pipeline is loaded
    public var limits: VoxtralMemoryLimits

    /// Maximum KV cache size (nil = unlimited, the default).
    /// Voxtral's LM has no sliding window: a limit does not evict the beginning of the audio, it makes
    /// generation throw `VoxtralError.contextTooLong` when prompt + `maxTokens` exceed it (K-2).
    public var maxKVCacheSize: Int?

    /// Bits of the quantized KV cache (group size 64); nil keeps it in the model's dtype (K-55)
    public var kvCacheBits: Int?

    /// Positions per prefill slice: attention scores grow with it (K-62)
    public var prefillStepSize: Int

    /// Backend `.mlx`: frees the audio tower's weights once the audio is encoded, and reads them again for the next
    /// request (K-63)
    public var releasesAudioTower: Bool

    /// `limits.cacheBytes`
    public var cacheLimitBytes: Int? {
        get { limits.cacheBytes }
        set { limits.cacheBytes = newValue }
    }

    public static let defaultPrefillStepSize = 512

    /// The ceiling measured on long audio: 19 013 MB of peak with Small on 8 min, against 37 313 MB without (K-34)
    public static let defaultCacheLimitBytes = 2048 * 1_048_576

    public init(
        limits: VoxtralMemoryLimits = .none,
        maxKVCacheSize: Int? = nil,
        kvCacheBits: Int? = nil,
        prefillStepSize: Int = Self.defaultPrefillStepSize,
        releasesAudioTower: Bool = false
    ) {
        self.limits = limits
        self.maxKVCacheSize = maxKVCacheSize
        self.kvCacheBits = kvCacheBits
        self.prefillStepSize = prefillStepSize
        self.releasesAudioTower = releasesAudioTower
    }

    // MARK: - Profiles

    /// Everything resident, no limit: the host's MLX settings stay
    public static var fast: MemoryOptimizationConfig { MemoryOptimizationConfig() }

    /// The lowest peak: limits sized from the available memory, 8-bit KV cache, small prefill slices, audio tower freed
    /// after the encoding. Sized when read: take it before loading.
    public static var lean: MemoryOptimizationConfig {
        MemoryOptimizationConfig(limits: .lean(), kvCacheBits: 8, prefillStepSize: 256, releasesAudioTower: true)
    }

    /// The default for a machine: `.fast`, with a 2 GB cache ceiling under 32 GB of RAM
    /// - Parameter ramGB: RAM in gigabytes (nil = this machine's)
    public static func recommended(forRAMGB ramGB: Int? = nil) -> MemoryOptimizationConfig {
        var config = MemoryOptimizationConfig.fast
        if ramGB ?? physicalRAMGB < 32 { config.cacheLimitBytes = defaultCacheLimitBytes }
        return config
    }

    static var physicalRAMGB: Int { Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)) }

    // MARK: - Description

    public var description: String {
        var parts: [String] = []
        if let bytes = limits.cacheBytes { parts.append("cacheLimit=\(bytes / 1_048_576)MB") }
        if let bytes = limits.memoryBytes { parts.append("memoryLimit=\(bytes / 1_048_576)MB") }
        if limits.clearsCacheBetweenStages { parts.append("clearBetweenStages") }
        if let maxKV = maxKVCacheSize { parts.append("maxKV=\(maxKV)") }
        if let bits = kvCacheBits { parts.append("kvBits=\(bits)") }
        parts.append("prefillStep=\(prefillStepSize)")
        if releasesAudioTower { parts.append("releaseAudioTower") }
        return "MemoryOptimizationConfig(\(parts.joined(separator: ", ")))"
    }
}
