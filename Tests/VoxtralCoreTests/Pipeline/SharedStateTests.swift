/**
 * SharedStateTests - K-16 (S-11)
 *
 * No global state mutated without a lock: each pipeline keeps its own memory
 * configuration (creating one no longer overwrites VoxtralMemoryManager.config), and the
 * shared globals (mel filter cache, memory manager, models directory, Hub API, debug flags)
 * survive concurrent access. Run under TSan:
 *   xcodebuild test … -enableThreadSanitizer YES -only-testing:VoxtralCoreTests/SharedStateTests
 *
 * Only APIs that existed before K-16 are used, so the suite compiles with the fix stashed.
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class SharedStateTests: XCTestCase {

    // MARK: - Memory configuration is carried by the pipeline

    func testTwoPipelinesKeepTheirOwnMemoryConfiguration() {
        let manager = VoxtralMemoryManager.shared
        let saved = manager.config
        defer { manager.config = saved }
        manager.config = .moderate

        let ultra = VoxtralPipeline(configuration: .init(memoryOptimization: .ultra))
        let disabled = VoxtralPipeline(configuration: .init(memoryOptimization: .disabled))

        XCTAssertEqual(ultra.configuration.memoryOptimization, .ultra)
        XCTAssertEqual(disabled.configuration.memoryOptimization, .disabled)
        XCTAssertEqual(manager.config, .moderate,
                       "creating a pipeline must not overwrite the shared configuration the other pipelines read")
    }

    // MARK: - Concurrent access to shared globals

    /// 4 log-mel extractions in parallel, each with a different mel count so that the
    /// filter cache is filled concurrently.
    func testFourConcurrentFeatureExtractions() {
        let melCounts = [128, 80, 64, 96]
        let lock = NSLock()
        nonisolated(unsafe) var bins: [Int: Int] = [:]
        DispatchQueue.concurrentPerform(iterations: melCounts.count) { i in
            let t = MLXArray(0 ..< 16_000).asType(.float32) / 16_000
            let audio = MLX.sin(t * Float(2 * Double.pi * 440))
            let (mel, _) = logMelSpectrogram(audio, nMels: melCounts[i])
            MLX.eval(mel)
            let count = mel.dim(0)
            lock.lock(); bins[melCounts[i]] = count; lock.unlock()
        }
        let result = bins
        for n in melCounts {
            XCTAssertEqual(result[n], n, "mel bins for nMels=\(n)")
        }
    }

    func testConcurrentSharedSettings() {
        let manager = VoxtralMemoryManager.shared
        let savedConfig = manager.config
        let savedDir = ModelDownloader.customModelsDirectory
        let savedDebug = VoxtralDebug.enabled
        defer {
            manager.config = savedConfig
            ModelDownloader.customModelsDirectory = savedDir
            VoxtralDebug.enabled = savedDebug
        }
        let presets: [MemoryOptimizationConfig] = [.disabled, .light, .moderate, .ultra]
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("voxtral-shared-state")

        DispatchQueue.concurrentPerform(iterations: 64) { i in
            manager.config = presets[i % presets.count]
            _ = manager.config.evalFrequency
            manager.optimizeIfNeeded(tokenIndex: i)
            manager.resetOptimizationCycle()
            ModelDownloader.customModelsDirectory = i.isMultiple(of: 2) ? dir : savedDir
            _ = ModelDownloader.customModelsDirectory
            _ = ModelDownloader.hubApi
            VoxtralDebug.enabled = false
            VoxtralDebug.log("concurrent \(i)")
        }
        XCTAssertTrue(presets.contains(manager.config))
    }
}
