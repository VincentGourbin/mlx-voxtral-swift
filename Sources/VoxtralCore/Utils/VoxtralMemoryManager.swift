/**
 * VoxtralMemoryManager - Centralized GPU memory management
 *
 * Provides a unified interface for memory operations inspired by flux-2-swift-mlx.
 * Includes monitoring, cleanup, and optimization utilities.
 */

import Foundation
import MLX

/// Centralized memory manager for Voxtral GPU operations
/// Thread-safe singleton for managing MLX GPU memory
public final class VoxtralMemoryManager: @unchecked Sendable {

    // MARK: - Singleton

    /// Shared instance for global memory management
    public static let shared = VoxtralMemoryManager()

    // MARK: - Properties

    /// Current memory optimization configuration
    public var config: MemoryOptimizationConfig = .recommended()

    /// Counter for tracking eval cycles (for periodic cleanup)
    private var evalCounter: Int = 0

    /// Lock for thread-safe operations
    private let lock = NSLock()

    // MARK: - Initialization

    private init() {}

    // MARK: - Memory Operations

    /// Clear the GPU cache to free unused memory
    /// Call this after large operations or when memory pressure is detected
    public func clearCache() {
        Memory.clearCache()
        VoxtralDebug.log("🧹 GPU cache cleared")
    }

    /// Full cleanup: clear cache and reset peak memory tracking
    /// Use this between transcription sessions for maximum memory recovery
    public func fullCleanup() {
        Memory.clearCache()
        GPU.resetPeakMemory()  // resetPeakMemory still on GPU
        evalCounter = 0
        VoxtralDebug.log("🧹 Full GPU cleanup performed")
    }

    /// Get current memory statistics
    /// - Returns: Tuple of (active memory bytes, cache memory bytes, peak memory bytes)
    public func memorySummary() -> (active: Int, cache: Int, peak: Int) {
        return (Memory.activeMemory, Memory.cacheMemory, Memory.peakMemory)
    }

    /// Get formatted memory summary string
    public func formattedMemorySummary() -> String {
        let (active, cache, peak) = memorySummary()
        return "GPU Memory: Active=\(formatBytes(active)), Cache=\(formatBytes(cache)), Peak=\(formatBytes(peak))"
    }

    /// Log current memory status
    public func logMemoryStatus() {
        VoxtralDebug.log(formattedMemorySummary())
    }

    // MARK: - Periodic Optimization

    /// Called during generation to apply memory optimization based on config
    /// - Parameter tokenIndex: Current token index in generation
    public func optimizeIfNeeded(tokenIndex: Int) {
        guard config.evalFrequency > 0 else { return }

        lock.lock()
        defer { lock.unlock() }

        evalCounter += 1

        if evalCounter >= config.evalFrequency {
            evalCounter = 0

            // Clear cache if configured
            if config.clearCacheOnEval {
                Memory.clearCache()
            }

            // Reset peak memory tracking if configured
            if config.resetPeakMemory {
                GPU.resetPeakMemory()
            }
        }
    }

    /// Reset the eval counter (call at start of new generation)
    public func resetOptimizationCycle() {
        lock.lock()
        evalCounter = 0
        lock.unlock()
    }

    // MARK: - Memory Warnings

    /// Check if memory usage is approaching critical levels
    /// - Parameter threshold: Percentage threshold (0.0-1.0) for warning
    /// - Returns: True if memory usage exceeds threshold
    public func isMemoryPressureHigh(threshold: Double = 0.8) -> Bool {
        let (active, cache, _) = memorySummary()
        let totalUsed = active + cache

        // Get system memory as reference
        let systemMemory = ProcessInfo.processInfo.physicalMemory
        let usageRatio = Double(totalUsed) / Double(systemMemory)

        return usageRatio > threshold
    }

    /// Perform emergency cleanup if memory pressure is high
    /// - Returns: True if cleanup was performed
    @discardableResult
    public func emergencyCleanupIfNeeded() -> Bool {
        if isMemoryPressureHigh(threshold: 0.9) {
            VoxtralDebug.always("⚠️ High memory pressure detected, performing emergency cleanup")
            fullCleanup()
            return true
        }
        return false
    }

    // MARK: - Utilities

    /// Format bytes as human-readable string
    private func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .memory
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

// MARK: - Convenience Extensions

extension VoxtralMemoryManager {

    /// Configure memory optimization based on available RAM
    public func autoConfigureForSystem() {
        config = .recommended()
        VoxtralDebug.log("Memory optimization auto-configured: \(config.description)")
    }

    /// Set memory optimization preset
    /// - Parameter preset: One of .disabled, .moderate, .aggressive, .ultra
    public func setPreset(_ preset: MemoryOptimizationConfig) {
        config = preset
        VoxtralDebug.log("Memory optimization set to: \(config.description)")
    }
}
