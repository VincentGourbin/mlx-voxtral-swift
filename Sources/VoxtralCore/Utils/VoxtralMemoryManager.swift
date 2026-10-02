/**
 * VoxtralMemoryManager - Centralized GPU memory management
 *
 * Provides a unified interface for memory operations inspired by flux-2-swift-mlx.
 * Includes monitoring, cleanup, and optimization utilities.
 */

import Foundation
import MLX

/// Centralized memory manager for Voxtral GPU operations
/// Singleton for managing MLX GPU memory; `config` and the eval counter are guarded by `lock`.
/// Pipelines pass their own configuration and never write `config` (S-11).
final class VoxtralMemoryManager: @unchecked Sendable {

    // MARK: - Singleton

    /// Shared instance for global memory management
    static let shared = VoxtralMemoryManager()

    // MARK: - Properties

    /// Default memory optimization configuration, used only when a caller passes none
    var config: MemoryOptimizationConfig {
        get { lock.lock(); defer { lock.unlock() }; return _config }
        set { lock.lock(); _config = newValue; lock.unlock() }
    }
    private var _config: MemoryOptimizationConfig = .recommended()

    /// Counter for tracking eval cycles (for periodic cleanup)
    private var evalCounter: Int = 0

    /// Lock for thread-safe operations
    private let lock = NSLock()

    // MARK: - Initialization

    private init() {}

    // MARK: - Memory Operations

    /// Clear the GPU cache to free unused memory
    /// Call this after large operations or when memory pressure is detected
    func clearCache() {
        Memory.clearCache()
        VoxtralDebug.log("🧹 GPU cache cleared")
    }

    /// Full cleanup: clear cache and reset the eval counter. The peak-memory counter is left alone:
    /// resetting it belongs to the measuring tool, not to the library (K-32, P-77)
    /// Use this between transcription sessions for maximum memory recovery
    func fullCleanup() {
        Memory.clearCache()
        lock.lock()
        evalCounter = 0
        lock.unlock()
        VoxtralDebug.log("🧹 Full GPU cleanup performed")
    }

    /// Get current memory statistics
    /// - Returns: Tuple of (active memory bytes, cache memory bytes, peak memory bytes)
    func memorySummary() -> (active: Int, cache: Int, peak: Int) {
        return (Memory.activeMemory, Memory.cacheMemory, Memory.peakMemory)
    }

    /// Get formatted memory summary string
    func formattedMemorySummary() -> String {
        let (active, cache, peak) = memorySummary()
        return "GPU Memory: Active=\(formatBytes(active)), Cache=\(formatBytes(cache)), Peak=\(formatBytes(peak))"
    }

    /// Log current memory status
    func logMemoryStatus() {
        VoxtralDebug.log(formattedMemorySummary())
    }

    // MARK: - Periodic Optimization

    /// Called during generation to apply memory optimization based on config
    /// - Parameter tokenIndex: Current token index in generation
    func optimizeIfNeeded(tokenIndex: Int) {
        optimizeIfNeeded(tokenIndex: tokenIndex, config: config)
    }

    /// Same, with the caller's configuration (a pipeline passes its own)
    func optimizeIfNeeded(tokenIndex: Int, config: MemoryOptimizationConfig) {
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
        }
    }

    /// Reset the eval counter (call at start of new generation)
    func resetOptimizationCycle() {
        lock.lock()
        evalCounter = 0
        lock.unlock()
    }

    // MARK: - Memory Warnings

    /// Check if memory usage is approaching critical levels
    /// - Parameter threshold: Percentage threshold (0.0-1.0) for warning
    /// - Returns: True if memory usage exceeds threshold
    func isMemoryPressureHigh(threshold: Double = 0.8) -> Bool {
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
    func emergencyCleanupIfNeeded() -> Bool {
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
    func autoConfigureForSystem() {
        config = .recommended()
        VoxtralDebug.log("Memory optimization auto-configured: \(config.description)")
    }

    /// Set memory optimization preset
    /// - Parameter preset: One of .disabled, .moderate, .aggressive, .ultra
    func setPreset(_ preset: MemoryOptimizationConfig) {
        config = preset
        VoxtralDebug.log("Memory optimization set to: \(config.description)")
    }
}
