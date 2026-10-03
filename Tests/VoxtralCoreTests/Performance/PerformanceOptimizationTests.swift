/**
 * PerformanceOptimizationTests - Unit tests for performance optimizations
 *
 * Covers: AVAudioConverter loading, top-p sampling, memory config presets,
 * and chunked prefill correctness.
 */

import XCTest
import MLX
@testable import VoxtralCore

final class PerformanceOptimizationTests: XCTestCase {

    // MARK: - MemoryOptimizationConfig Tests

    // K-2: Voxtral's LM has no sliding window, so no preset limits the KV cache; presets differ by
    // their eval / cache-clearing rhythm.
    func testPresetsHaveNoKVCacheLimit() {
        for config in [MemoryOptimizationConfig.light, .moderate, .aggressive, .ultra, .disabled] {
            XCTAssertNil(config.maxKVCacheSize, "\(config.description)")
        }
        XCTAssertEqual(MemoryOptimizationConfig.light.evalFrequency, 16)
        XCTAssertEqual(MemoryOptimizationConfig.ultra.evalFrequency, 2)
        XCTAssertTrue(MemoryOptimizationConfig.aggressive.clearCacheOnEval)
    }

    func testDisabledPresetHasNoKVCacheLimit() {
        let config = MemoryOptimizationConfig.disabled
        XCTAssertNil(config.maxKVCacheSize)
        XCTAssertEqual(config.evalFrequency, 0)
    }

    func testRecommendedNeverReturnsDisabled() {
        // For any RAM size, recommended() returns a config that evaluates periodically
        for ram in [8, 16, 32, 64, 128] {
            XCTAssertGreaterThan(MemoryOptimizationConfig.recommended(forRAMGB: ram).evalFrequency, 0, "\(ram) GB")
        }
    }

    func testRecommendedPresetScaling() {
        let config8 = MemoryOptimizationConfig.recommended(forRAMGB: 8)
        let config16 = MemoryOptimizationConfig.recommended(forRAMGB: 16)
        let config64 = MemoryOptimizationConfig.recommended(forRAMGB: 64)

        // Less RAM → more frequent evaluation
        XCTAssertLessThan(config8.evalFrequency, config16.evalFrequency)
        XCTAssertLessThan(config16.evalFrequency, config64.evalFrequency)
    }

    func testRecommendedAutoDetection() {
        // Should not crash and return a valid config
        let config = MemoryOptimizationConfig.recommended()
        XCTAssertGreaterThan(config.evalFrequency, 0, "Auto-detected config should have eval frequency > 0")
        XCTAssertNil(config.maxKVCacheSize, "no preset limits the KV cache (K-2)")
    }

    // MARK: - Audio Loading Tests

    func testLoadAudioThrowsForEmptyPath() {
        XCTAssertThrowsError(try loadAudio("")) { error in
            XCTAssertNotNil(error)
        }
    }

    func testLoadAudioThrowsForNonexistentFile() {
        XCTAssertThrowsError(try loadAudio("/nonexistent/audio.wav")) { error in
            XCTAssertNotNil(error)
        }
    }

    // MARK: - Sampling filter (production `nucleusMask`, K-29)

    /// Logits 0, 1, …, n-1: the most probable tokens are the last ones
    private func ramp(_ n: Int) -> MLXArray {
        MLXArray((0 ..< n).map { Float($0) / 50 }).reshaped(1, n)
    }

    private func keptCount(_ masked: MLXArray) -> Int {
        (masked .> MLXArray(-Float.infinity)).sum().item(Int.self)
    }

    func testNucleusMaskKeepsTheCandidatesWhenTheirMassReachesTopP() {
        let logits = ramp(1500)
        let masked = VoxtralForConditionalGeneration.nucleusMask(logits, topP: 0.9)
        XCTAssertEqual(keptCount(masked), 1000, "the 1000 most probable tokens hold ≥ 90 % of the mass")
        XCTAssertEqual(masked[0, 1499].item(Float.self), logits[0, 1499].item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(masked[0, 0].item(Float.self), -Float.infinity, "the least probable token is masked")
    }

    func testNucleusMaskKeepsEverythingWhenTheCandidatesMissTopP() {
        // Increasing logits: the top 1000 hold ≈ 0.81 of the mass, below topP 0.9 → nothing may be masked (K-29: the
        // former uniform input passed even with the kth-probability cutoff)
        let ramped = MLXArray((0 ..< 1500).map { Float($0) / 1000 }).reshaped(1, 1500)
        XCTAssertEqual(keptCount(VoxtralForConditionalGeneration.nucleusMask(ramped, topP: 0.9)), 1500)
    }

    func testNucleusMaskDoesNotFilterASmallVocabulary() {
        // Below `candidates` tokens the candidates are the whole vocabulary: nothing is masked (not an exact nucleus)
        XCTAssertEqual(keptCount(VoxtralForConditionalGeneration.nucleusMask(ramp(10), topP: 0.5)), 10)
    }

    // MARK: - Chunked prefill (production `prefillChunkRanges`, K-29)

    func testPrefillChunksCoverTheSequenceWithARemainder() {
        XCTAssertEqual(VoxtralForConditionalGeneration.prefillChunkRanges(totalLength: 700, chunkSize: 512),
                       [0 ..< 512, 512 ..< 700])
    }

    func testPrefillChunksOfAnExactMultiple() {
        XCTAssertEqual(VoxtralForConditionalGeneration.prefillChunkRanges(totalLength: 1024, chunkSize: 512),
                       [0 ..< 512, 512 ..< 1024])
    }

    func testPrefillChunksOfAShortSequence() {
        XCTAssertEqual(VoxtralForConditionalGeneration.prefillChunkRanges(totalLength: 256, chunkSize: 512), [0 ..< 256])
    }

    // MARK: - TTS end-of-audio batch check (production `shouldCheckEOA` / `firstEOA`, K-29)

    func testEOACheckScheduleCoversEveryFrame() {
        let interval = 4, total = 17
        var covered = Set<Int>()
        for frame in 0 ..< total where VoxtralTTSModel.shouldCheckEOA(frame: frame, interval: interval) {
            covered.formUnion(max(0, frame + 1 - interval) ... frame)
        }
        XCTAssertEqual(covered, Set(0 ..< 16), "every frame up to the last check boundary is checked once")
        let checks = (0 ..< total).filter { VoxtralTTSModel.shouldCheckEOA(frame: $0, interval: interval) }
        XCTAssertEqual(checks, [0, 3, 7, 11, 15], "the first frame is checked at once (an immediate end of audio)")
    }

    func testFirstEOAFindsTheEndFrame() {
        XCTAssertEqual(VoxtralTTSModel.firstEOA(in: [10, 20, 0, 30]), 2)
        XCTAssertEqual(VoxtralTTSModel.firstEOA(in: [10, 1, 0]), 1, "codes 0 and 1 both end the audio")
    }

    func testFirstEOAAtTheFirstFrame() {
        XCTAssertEqual(VoxtralTTSModel.firstEOA(in: [0, 10, 20]), 0)
    }

    func testFirstEOAWithoutEnd() {
        XCTAssertNil(VoxtralTTSModel.firstEOA(in: [10, 20, 30]))
    }
}

