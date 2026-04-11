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

    func testLightPresetHasKVCacheLimit() {
        let config = MemoryOptimizationConfig.light
        XCTAssertNotNil(config.maxKVCacheSize, "Light preset should have KV cache limit")
        XCTAssertEqual(config.maxKVCacheSize, 8192)
        XCTAssertEqual(config.evalFrequency, 16)
    }

    func testModeratePresetHasKVCacheLimit() {
        let config = MemoryOptimizationConfig.moderate
        XCTAssertNotNil(config.maxKVCacheSize, "Moderate preset should have KV cache limit")
        XCTAssertEqual(config.maxKVCacheSize, 6144)
    }

    func testAggressivePresetHasKVCacheLimit() {
        let config = MemoryOptimizationConfig.aggressive
        XCTAssertEqual(config.maxKVCacheSize, 4096)
        XCTAssertTrue(config.clearCacheOnEval)
    }

    func testUltraPresetHasKVCacheLimit() {
        let config = MemoryOptimizationConfig.ultra
        XCTAssertEqual(config.maxKVCacheSize, 2048)
        XCTAssertEqual(config.evalFrequency, 2)
    }

    func testDisabledPresetHasNoKVCacheLimit() {
        let config = MemoryOptimizationConfig.disabled
        XCTAssertNil(config.maxKVCacheSize)
        XCTAssertEqual(config.evalFrequency, 0)
    }

    func testRecommendedNeverReturnsDisabled() {
        // For any RAM size, recommended() should return a config with KV cache limit
        let config8 = MemoryOptimizationConfig.recommended(forRAMGB: 8)
        let config16 = MemoryOptimizationConfig.recommended(forRAMGB: 16)
        let config32 = MemoryOptimizationConfig.recommended(forRAMGB: 32)
        let config64 = MemoryOptimizationConfig.recommended(forRAMGB: 64)
        let config128 = MemoryOptimizationConfig.recommended(forRAMGB: 128)

        XCTAssertNotNil(config8.maxKVCacheSize)
        XCTAssertNotNil(config16.maxKVCacheSize)
        XCTAssertNotNil(config32.maxKVCacheSize)
        XCTAssertNotNil(config64.maxKVCacheSize)
        XCTAssertNotNil(config128.maxKVCacheSize)
    }

    func testRecommendedPresetScaling() {
        let config8 = MemoryOptimizationConfig.recommended(forRAMGB: 8)
        let config16 = MemoryOptimizationConfig.recommended(forRAMGB: 16)
        let config64 = MemoryOptimizationConfig.recommended(forRAMGB: 64)

        // More RAM → larger KV cache limit
        XCTAssertLessThan(config8.maxKVCacheSize!, config16.maxKVCacheSize!)
        XCTAssertLessThan(config16.maxKVCacheSize!, config64.maxKVCacheSize!)
    }

    func testRecommendedAutoDetection() {
        // Should not crash and return a valid config
        let config = MemoryOptimizationConfig.recommended()
        XCTAssertGreaterThan(config.evalFrequency, 0, "Auto-detected config should have eval frequency > 0")
        XCTAssertNotNil(config.maxKVCacheSize)
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

    // MARK: - Top-p Sampling Logic Tests (via MLX operations)

    func testTopKPartialSort() {
        // Verify top() returns k largest values
        let logits = MLXArray([1.0, 5.0, 3.0, 7.0, 2.0, 9.0, 4.0, 6.0, 8.0, 0.0] as [Float])
        let topK = top(logits, k: 3)
        eval(topK)

        XCTAssertEqual(topK.shape[0], 3)
        // Top 3 should be 9, 8, 7 (unsorted)
        let values = (0..<3).map { topK[$0].item(Float.self) }
        let sortedValues = values.sorted(by: >)
        XCTAssertEqual(sortedValues[0], 9.0, accuracy: 1e-5)
        XCTAssertEqual(sortedValues[1], 8.0, accuracy: 1e-5)
        XCTAssertEqual(sortedValues[2], 7.0, accuracy: 1e-5)
    }

    func testSoftmaxSumsToOne() {
        let logits = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0] as [Float]).reshaped(1, 5)
        let probs = softmax(logits, axis: -1)
        eval(probs)

        let sum = probs.sum().item(Float.self)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5)
    }

    func testTopPMaskingPreservesHighProbTokens() {
        // Simulate top-p masking: tokens above cutoff should survive
        let probs = MLXArray([0.5, 0.3, 0.1, 0.05, 0.05] as [Float]).reshaped(1, 5)
        let logits = MLXArray([2.0, 1.5, 0.5, 0.0, 0.0] as [Float]).reshaped(1, 5)
        let cutoff = MLXArray(Float(0.1))

        let masked = which(probs .>= cutoff, logits, -Float.infinity)
        eval(masked)

        // Tokens with prob >= 0.1: indices 0 (0.5), 1 (0.3), 2 (0.1)
        let val0 = masked[0, 0].item(Float.self)
        let val1 = masked[0, 1].item(Float.self)
        let val2 = masked[0, 2].item(Float.self)
        let val3 = masked[0, 3].item(Float.self)

        XCTAssertEqual(val0, 2.0, accuracy: 1e-5, "High prob token should survive")
        XCTAssertEqual(val1, 1.5, accuracy: 1e-5, "High prob token should survive")
        XCTAssertEqual(val2, 0.5, accuracy: 1e-5, "Borderline token should survive")
        XCTAssertEqual(val3, -Float.infinity, "Low prob token should be masked")
    }

    // MARK: - Chunked Prefill Logic Tests

    func testEmbeddingSlicingPreservesShape() {
        // Verify that slicing embeddings for chunked prefill preserves dimensions
        let seqLen = 1024
        let hiddenSize = 128
        let embeddings = MLXArray.ones([1, seqLen, hiddenSize])
        let chunkSize = 512

        let chunk1 = embeddings[0..., 0..<chunkSize, 0...]
        let chunk2 = embeddings[0..., chunkSize..<seqLen, 0...]
        eval(chunk1, chunk2)

        XCTAssertEqual(chunk1.shape, [1, 512, 128], "Chunk 1 should have correct shape")
        XCTAssertEqual(chunk2.shape, [1, 512, 128], "Chunk 2 should have correct shape")
    }

    func testEmbeddingSlicingPreservesValues() {
        let embeddings = MLXArray(Array(stride(from: Float(0), to: Float(12), by: 1))).reshaped(1, 4, 3)

        let chunk1 = embeddings[0..., 0..<2, 0...]
        let chunk2 = embeddings[0..., 2..<4, 0...]
        eval(chunk1, chunk2)

        // chunk1 should be [[0,1,2],[3,4,5]]
        XCTAssertEqual(chunk1[0, 0, 0].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(chunk1[0, 1, 2].item(Float.self), 5.0, accuracy: 1e-5)

        // chunk2 should be [[6,7,8],[9,10,11]]
        XCTAssertEqual(chunk2[0, 0, 0].item(Float.self), 6.0, accuracy: 1e-5)
        XCTAssertEqual(chunk2[0, 1, 2].item(Float.self), 11.0, accuracy: 1e-5)
    }

    func testChunkedPrefillChunkSizeHandlesRemainder() {
        // Verify stride handles non-divisible sequence lengths
        let totalSeqLen = 700
        let chunkSize = 512

        var chunks: [(start: Int, end: Int)] = []
        for chunkStart in stride(from: 0, to: totalSeqLen, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, totalSeqLen)
            chunks.append((chunkStart, chunkEnd))
        }

        XCTAssertEqual(chunks.count, 2, "Should have 2 chunks")
        XCTAssertEqual(chunks[0].start, 0)
        XCTAssertEqual(chunks[0].end, 512)
        XCTAssertEqual(chunks[1].start, 512)
        XCTAssertEqual(chunks[1].end, 700)
    }

    func testChunkedPrefillSingleChunkFallback() {
        // Sequences shorter than chunk size should use single pass
        let totalSeqLen = 256
        let chunkSize = 512

        var chunkCount = 0
        if totalSeqLen > chunkSize {
            for _ in stride(from: 0, to: totalSeqLen, by: chunkSize) {
                chunkCount += 1
            }
        }

        XCTAssertEqual(chunkCount, 0, "Short sequences should not enter chunk loop")
    }

    // MARK: - TTS EOA Batch Check Tests

    func testEOACheckIntervalCoversAllFrames() {
        let eoaCheckInterval = 4
        let totalFrames = 17

        var checkedFrames: [Int] = []
        for i in 0..<totalFrames {
            if (i + 1) % eoaCheckInterval == 0 || i == 0 {
                let checkStart = max(0, (i + 1) - eoaCheckInterval)
                for j in checkStart...(i) {
                    checkedFrames.append(j)
                }
            }
        }

        // All frames up to the last check boundary should be covered
        let lastCheckBoundary = ((totalFrames) / eoaCheckInterval) * eoaCheckInterval - 1
        for frame in 0...lastCheckBoundary {
            XCTAssertTrue(checkedFrames.contains(frame), "Frame \(frame) should be checked")
        }
    }

    func testEOATrimRemovesPostEOAFrames() {
        // Simulate: frames [valid, valid, EOA, valid] → should trim to [valid, valid]
        var allCodes: [Int] = [10, 20, 0, 30]  // 0 = EOA

        for j in 0..<allCodes.count {
            if allCodes[j] <= 1 {
                allCodes = Array(allCodes.prefix(j))
                break
            }
        }

        XCTAssertEqual(allCodes, [10, 20], "Should trim at EOA frame")
    }

    func testEOATrimAtFirstFrame() {
        var allCodes: [Int] = [0, 10, 20]

        for j in 0..<allCodes.count {
            if allCodes[j] <= 1 {
                allCodes = Array(allCodes.prefix(j))
                break
            }
        }

        XCTAssertEqual(allCodes, [], "EOA at frame 0 should produce empty result")
    }
}

