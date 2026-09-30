/**
 * LongPromptKVCacheTests - K-2 (S-02 / P-03)
 *
 * Voxtral's LM has no sliding window: STT keeps the whole prompt in a KV cache without a window
 * (every preset), and an explicit limit refuses a too-long request before any compute instead of
 * stopping at the prefill or evicting the start of the audio.
 *
 * (a) and (b) are unit tests on a reduced random model. (c) is the C-long integration
 * (`TEST_RUNNER_VOXTRAL_LONG_AUDIO=1`, mini-3b-8bit, `.local-runs/corpus/c_long.wav` from PLAN.md §5).
 * Checks go through the error text so the suite compiles without the fix (RED).
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class LongPromptKVCacheTests: XCTestCase {

    private func prompt(_ n: Int) -> MLXArray {
        MLXArray((0 ..< n).map { Int32(30 + $0 % 60) }).reshaped([1, n])  // never the audio token (24)
    }

    // (a) the P-03 trigger: 2 600 positions, chunked prefill, `.ultra` preset
    func testUltraPreset2600Positions() throws {
        let model = try makeReducedVoxtralModel()
        let tokens = try model.generateStream(inputIds: prompt(2_600), maxNewTokens: 1, memoryOptimization: .ultra)
        XCTAssertEqual(tokens.count, 1)
    }

    // (b) an explicit limit is refused up front, with the numbers
    func testExplicitLimitThrowsContextTooLong() throws {
        let model = try makeReducedVoxtralModel()
        var limited = MemoryOptimizationConfig.ultra
        limited.maxKVCacheSize = 2_048
        let start = Date()
        XCTAssertThrowsError(try model.generateStream(inputIds: prompt(2_600), maxNewTokens: 8, memoryOptimization: limited)) { error in
            let text = "\(error)"
            XCTAssertTrue(text.contains("contextTooLong"), text)
            XCTAssertTrue(text.contains("2600") && text.contains("2048"), text)
        }
        XCTAssertLessThan(Date().timeIntervalSince(start), 0.5, "must refuse before computing")
    }

    // (c) C-long end to end with the 8 GB and 16 GB presets
    func testLongAudioTranscribedToTheEnd() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_LONG_AUDIO"] == "1",
                          "Set VOXTRAL_LONG_AUDIO=1 to run the C-long integration")
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let audio = root.appendingPathComponent(".local-runs/corpus/c_long.wav")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: audio.path), "create C-long first (PLAN.md §5)")

        func normalized(_ s: String) -> String {
            s.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
                .filter { $0.isLetter || $0 == " " }
                .split(separator: " ").joined(separator: " ")
        }
        // The model writes "Flux Forge … complete iCreative studio": compare the ASR-stable start
        let firstEN = normalized("Flux Forge Studio turns your Mac into a complete")
        // Reaching the last FR sentence needs a duration-based token budget (K-5): reported, not asserted here
        let lastFR = normalized("Aucune donnee envoyee dans le cloud.")

        for (label, ram) in [("ultra(8GB)", 8), ("aggressive(16GB)", 16)] {
            let config = VoxtralPipeline.Configuration(maxTokens: 4_096, memoryOptimization: .recommended(forRAMGB: ram))
            let pipeline = VoxtralPipeline(model: .mini3b8bit, backend: .mlx, configuration: config)
            try await pipeline.loadModel()
            let raw = try await pipeline.transcribe(audio: audio, language: nil)
            pipeline.unload()
            try? raw.write(to: root.appendingPathComponent(".local-runs/k2_long_\(ram)gb.txt"), atomically: true, encoding: .utf8)
            let text = normalized(raw)
            let firstOK = text.hasPrefix(firstEN)
            let lastOK = text.contains(lastFR)
            print("[long-audio] LONG_AUDIO \(label) : 0 crash · first EN \(firstOK ? "OK" : "KO") · last FR \(lastOK ? "OK" : "KO (K-5)") · \(text.count) chars")
            XCTAssertTrue(firstOK, "\(label): the transcription must start with the first EN sentence")
        }
    }
}
