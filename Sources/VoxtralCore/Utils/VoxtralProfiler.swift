/**
 * VoxtralProfiler - Detailed profiling for inference pipeline
 *
 * Tracks memory (MLX GPU + system) and time for each step.
 * Uses MLX.GPU for GPU memory and mach task_info for process memory.
 *
 * References:
 * - MLX GPU: https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/GPU.swift
 * - task_vm_info: https://developer.apple.com/forums/thread/105088
 * - phys_footprint: https://gist.github.com/pejalo/671dd2f67e3877b18c38c749742350ca
 */

import Foundation
import MLX

// MARK: - Memory Snapshot

/// Complete memory snapshot combining MLX GPU and system memory
public struct MemorySnapshot: CustomStringConvertible, Sendable {
    /// MLX active memory (in MLXArrays)
    public let mlxActive: Int
    /// MLX cache memory (recyclable)
    public let mlxCache: Int
    /// MLX peak memory since last reset
    public let mlxPeak: Int
    /// Process physical footprint (most accurate for "real" memory)
    public let processFootprint: Int64
    /// Timestamp
    public let timestamp: Date

    public var mlxTotal: Int { mlxActive + mlxCache }

    public var description: String {
        """
        MLX Active: \(formatBytes(mlxActive)) | Cache: \(formatBytes(mlxCache)) | Peak: \(formatBytes(mlxPeak))
        Process: \(formatBytes(Int(processFootprint)))
        """
    }

    /// Compute delta between two snapshots
    public func delta(to other: MemorySnapshot) -> MemoryDelta {
        MemoryDelta(
            mlxActiveDelta: other.mlxActive - mlxActive,
            mlxCacheDelta: other.mlxCache - mlxCache,
            mlxPeakDelta: other.mlxPeak - mlxPeak,
            processFootprintDelta: other.processFootprint - processFootprint,
            duration: other.timestamp.timeIntervalSince(timestamp)
        )
    }
}

/// Delta between two memory snapshots
public struct MemoryDelta: CustomStringConvertible, Sendable {
    public let mlxActiveDelta: Int
    public let mlxCacheDelta: Int
    public let mlxPeakDelta: Int
    public let processFootprintDelta: Int64
    public let duration: TimeInterval

    public var description: String {
        let sign = { (v: Int) -> String in v >= 0 ? "+" : "" }
        let sign64 = { (v: Int64) -> String in v >= 0 ? "+" : "" }
        return """
        MLX: \(sign(mlxActiveDelta))\(formatBytes(mlxActiveDelta)) active, \(sign(mlxCacheDelta))\(formatBytes(mlxCacheDelta)) cache
        Process: \(sign64(processFootprintDelta))\(formatBytes(Int(processFootprintDelta)))
        Duration: \(String(format: "%.3f", duration))s
        """
    }
}

// MARK: - Step Result

/// Result of a profiled step
public struct ProfiledStep: CustomStringConvertible, Sendable {
    public let name: String
    public let startMemory: MemorySnapshot
    public let endMemory: MemorySnapshot
    public let duration: TimeInterval

    public init(name: String, startMemory: MemorySnapshot, endMemory: MemorySnapshot, duration: TimeInterval) {
        self.name = name
        self.startMemory = startMemory
        self.endMemory = endMemory
        self.duration = duration
    }

    public var delta: MemoryDelta {
        startMemory.delta(to: endMemory)
    }

    public var description: String {
        """
        [\(name)] \(String(format: "%.3f", duration))s
          Start: MLX \(formatBytes(startMemory.mlxActive)) | Process \(formatBytes(Int(startMemory.processFootprint)))
          End:   MLX \(formatBytes(endMemory.mlxActive)) | Process \(formatBytes(Int(endMemory.processFootprint)))
          Delta: MLX \(formatDeltaBytes(endMemory.mlxActive - startMemory.mlxActive)) | Process \(formatDeltaBytes(Int(endMemory.processFootprint - startMemory.processFootprint)))
        """
    }
}

// MARK: - Profiler

/// Main profiler class for tracking inference pipeline
public class VoxtralProfiler {

    /// All recorded steps
    public var steps: [ProfiledStep] = []

    /// Initial memory snapshot
    public private(set) var initialSnapshot: MemorySnapshot?

    /// Device info
    public let deviceInfo: GPU.DeviceInfo

    public init() {
        self.deviceInfo = GPU.deviceInfo()
    }

    /// Take a memory snapshot
    public static func snapshot() -> MemorySnapshot {
        MemorySnapshot(
            mlxActive: Memory.activeMemory,
            mlxCache: Memory.cacheMemory,
            mlxPeak: Memory.peakMemory,
            processFootprint: getProcessMemoryFootprint(),
            timestamp: Date()
        )
    }

    /// Start profiling session (resets peak memory)
    public func start() {
        steps.removeAll()
        GPU.resetPeakMemory()
        initialSnapshot = Self.snapshot()
    }

    /// Profile a synchronous step
    public func profile<T>(_ name: String, _ block: () throws -> T) rethrows -> T {
        let startMemory = Self.snapshot()
        let startTime = Date()

        let result = try block()

        // Force evaluation of any lazy MLX operations
        eval()

        let endTime = Date()
        let endMemory = Self.snapshot()

        let step = ProfiledStep(
            name: name,
            startMemory: startMemory,
            endMemory: endMemory,
            duration: endTime.timeIntervalSince(startTime)
        )
        steps.append(step)

        return result
    }

    /// Profile an async step
    public func profileAsync<T>(_ name: String, _ block: () async throws -> T) async rethrows -> T {
        let startMemory = Self.snapshot()
        let startTime = Date()

        let result = try await block()

        // Force evaluation of any lazy MLX operations
        eval()

        let endTime = Date()
        let endMemory = Self.snapshot()

        let step = ProfiledStep(
            name: name,
            startMemory: startMemory,
            endMemory: endMemory,
            duration: endTime.timeIntervalSince(startTime)
        )
        steps.append(step)

        return result
    }

    /// Get summary report
    public func summary() -> ProfileSummary {
        let finalSnapshot = Self.snapshot()
        return ProfileSummary(
            deviceInfo: deviceInfo,
            initialSnapshot: initialSnapshot ?? finalSnapshot,
            finalSnapshot: finalSnapshot,
            steps: steps
        )
    }

    /// Clear MLX cache and take new snapshot
    public func clearCacheAndSnapshot() -> MemorySnapshot {
        Memory.clearCache()
        return Self.snapshot()
    }
}

// MARK: - Profile Summary

/// Complete profiling summary
public struct ProfileSummary: CustomStringConvertible {
    public let deviceInfo: GPU.DeviceInfo
    public let initialSnapshot: MemorySnapshot
    public let finalSnapshot: MemorySnapshot
    public let steps: [ProfiledStep]

    public var totalDuration: TimeInterval {
        steps.reduce(0) { $0 + $1.duration }
    }

    public var totalMemoryGrowth: Int {
        finalSnapshot.mlxActive - initialSnapshot.mlxActive
    }

    public var peakMemoryUsed: Int {
        finalSnapshot.mlxPeak
    }

    public var description: String {
        var lines: [String] = []

        lines.append("=" .repeated(60))
        lines.append("VOXTRAL PROFILING SUMMARY")
        lines.append("=" .repeated(60))

        lines.append("")
        lines.append("Device: \(deviceInfo.architecture)")
        lines.append("System RAM: \(formatBytes(deviceInfo.memorySize))")
        lines.append("Recommended Working Set: \(formatBytes(Int(deviceInfo.maxRecommendedWorkingSetSize)))")

        lines.append("")
        lines.append("-" .repeated(60))
        lines.append("STEPS")
        lines.append("-" .repeated(60))

        for step in steps {
            let memDelta = step.endMemory.mlxActive - step.startMemory.mlxActive
            let procDelta = step.endMemory.processFootprint - step.startMemory.processFootprint
            lines.append(String(format: "%-25s %8.3fs  MLX: %+10s  Proc: %+10s",
                               (step.name as NSString).utf8String!,
                               step.duration,
                               (formatDeltaBytes(memDelta) as NSString).utf8String!,
                               (formatDeltaBytes(Int(procDelta)) as NSString).utf8String!))
        }

        lines.append("")
        lines.append("-" .repeated(60))
        lines.append("TOTALS")
        lines.append("-" .repeated(60))
        lines.append("Total Duration: \(String(format: "%.3f", totalDuration))s")
        lines.append("MLX Peak Memory: \(formatBytes(peakMemoryUsed))")
        lines.append("MLX Final Active: \(formatBytes(finalSnapshot.mlxActive))")
        lines.append("MLX Final Cache: \(formatBytes(finalSnapshot.mlxCache))")
        lines.append("Process Footprint: \(formatBytes(Int(finalSnapshot.processFootprint)))")

        lines.append("")
        lines.append("=" .repeated(60))

        return lines.joined(separator: "\n")
    }
}

// MARK: - System Memory

/// Get process physical memory footprint using task_vm_info
/// This is the most accurate measure of "real" memory usage
private func getProcessMemoryFootprint() -> Int64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)

    let result = withUnsafeMutablePointer(to: &info) { infoPtr in
        infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
        }
    }

    if result == KERN_SUCCESS {
        return Int64(info.phys_footprint)
    }
    return 0
}

// MARK: - Formatting Helpers

private func formatBytes(_ bytes: Int) -> String {
    let absBytes = abs(bytes)
    if absBytes >= 1024 * 1024 * 1024 {
        return String(format: "%.2f GB", Double(bytes) / (1024 * 1024 * 1024))
    } else if absBytes >= 1024 * 1024 {
        return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
    } else if absBytes >= 1024 {
        return String(format: "%.1f KB", Double(bytes) / 1024)
    }
    return "\(bytes) B"
}

private func formatDeltaBytes(_ bytes: Int) -> String {
    let sign = bytes >= 0 ? "+" : ""
    return sign + formatBytes(bytes)
}

private extension String {
    func repeated(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}
