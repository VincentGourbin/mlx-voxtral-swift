/**
 * VoxtralMemoryManager - MLX memory limits sized for the machine, and what is in use
 *
 * Replaces the `light` to `ultra` presets, which only changed how often the decoding loop flushed (K-51). Same pattern
 * as YuE2MemoryManager: a ceiling on MLX's buffer cache per stage, and `lean` limits sized from what the system leaves
 * the process: YuE2's first fixed mobile limits cost +32 % of time in 4 bits and +73 % in bf16.
 */

import Foundation
import MLX
#if os(iOS)
import os
#endif

private let mebibyte = 1_048_576

/// The MLX limits a pipeline holds while it is loaded. They are process-wide: `nil` leaves the host's setting alone, and
/// the pipeline gives back what it changed at `unload()`.
public struct VoxtralMemoryLimits: Sendable, Equatable {
    /// Ceiling of MLX's buffer cache (`Memory.cacheLimit`). Without one, MLX keeps every freed buffer: 70 GB of process
    /// for 8.6 GB of active memory on 11 min of audio (K-52)
    public var cacheBytes: Int?
    /// `Memory.memoryLimit`: above it MLX empties its cache and makes each evaluation wait for the work in progress, so
    /// keep it above the weights. Not a hard ceiling
    public var memoryBytes: Int?
    /// Empties the cache between two stages of a request (encoder then decoder, decoder then codec)
    public var clearsCacheBetweenStages: Bool

    public init(cacheBytes: Int? = nil, memoryBytes: Int? = nil, clearsCacheBetweenStages: Bool = false) {
        self.cacheBytes = cacheBytes
        self.memoryBytes = memoryBytes
        self.clearsCacheBetweenStages = clearsCacheBetweenStages
    }

    /// No limit: the host's settings stay
    public static let none = VoxtralMemoryLimits()

    /// Sized from what the process can still allocate: a cache of `available / 6` between 256 MB and 1 GB and, on iOS, a
    /// memory threshold of `available` − 1.25 GB, never below 3 GB. Not on a Mac: `available` is a fixed figure there, and
    /// a pack heavier than the threshold would make every evaluation wait. Read it before the weights load.
    public static func lean(availableBytes: Int = VoxtralMemoryManager.availableBytes) -> VoxtralMemoryLimits {
        var limits = VoxtralMemoryLimits(
            cacheBytes: min(1024 * mebibyte, max(256 * mebibyte, availableBytes / 6)), clearsCacheBetweenStages: true)
        #if os(iOS)
        limits.memoryBytes = leanMemoryBytes(availableBytes: availableBytes)
        #endif
        return limits
    }

    static func leanMemoryBytes(availableBytes: Int) -> Int {
        max(3072 * mebibyte, availableBytes - 1280 * mebibyte)
    }

    /// The stages of a pipeline's life that have their own ceiling or start with a boundary
    enum Stage: Sendable { case load, generate, codec }

    /// Cache ceiling of `stage`: loading reuses few buffers, so it stays under 1 GB whatever the ceiling
    func cacheBytes(for stage: Stage) -> Int? {
        guard let cacheBytes else { return nil }
        return stage == .load ? min(cacheBytes, 1024 * mebibyte) : cacheBytes
    }
}

/// What a process uses at one moment
public struct VoxtralMemoryReport: Sendable, CustomStringConvertible {
    /// MLX buffers in use
    public let activeBytes: Int
    /// MLX buffers kept for reuse
    public let cacheBytes: Int
    /// Highest `activeBytes` since the last reset by a measuring tool
    public let peakBytes: Int
    /// `phys_footprint`, what iOS counts against the process's limit; nil if the system refuses to tell
    public let footprintBytes: Int?
    /// See `VoxtralMemoryManager.availableBytes`
    public let availableBytes: Int

    public var description: String {
        let footprint = footprintBytes.map { ", footprint \(Self.format($0))" } ?? ""
        return "GPU memory: active \(Self.format(activeBytes)), cache \(Self.format(cacheBytes)), "
            + "peak \(Self.format(peakBytes))\(footprint), available \(Self.format(availableBytes))"
    }

    private static func format(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
}

/// Sizes the limits and reports usage. The limits themselves live in each pipeline's configuration.
public enum VoxtralMemoryManager {

    /// What the process can still allocate: `os_proc_available_memory` on iOS. On a Mac, 6 GB, the figure of an 8 GB
    /// iPhone, so that a `lean` measured on a Mac runs under the cache ceiling it will have on the phone.
    public static var availableBytes: Int {
        #if os(iOS)
        return Int(os_proc_available_memory())
        #else
        return 6 * 1024 * mebibyte
        #endif
    }

    /// Current memory usage
    public static func report() -> VoxtralMemoryReport {
        let snapshot = Memory.snapshot()
        return VoxtralMemoryReport(
            activeBytes: snapshot.activeMemory, cacheBytes: snapshot.cacheMemory, peakBytes: snapshot.peakMemory,
            footprintBytes: footprintBytes(), availableBytes: availableBytes)
    }

    private static func footprintBytes() -> Int? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) : nil
    }
}
