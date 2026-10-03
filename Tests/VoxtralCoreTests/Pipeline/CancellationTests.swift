/**
 * CancellationTests - K-15 (cooperative cancellation, work off the cooperative pool)
 *
 * A cancelled Task stops a long generation within 2 s with CancellationError and leaves the pipeline `.ready`:
 * STT on C-long (≈ 11 min, mini-3b-8bit, backends `.mlx` and `.auto` with Core ML), TTS batch on a long text (tts-4b-4bit), Realtime on C-long.
 * Each run is cancelled 3 s after it starts. Before K-15 the work ran to its end (minutes).
 * Heavy (real models): skipped unless VOXTRAL_CANCEL=1 (`TEST_RUNNER_VOXTRAL_CANCEL=1 xcodebuild test …`);
 * C-long is `.local-runs/corpus/c_long.wav` (PLAN.md §5).
 */

import Foundation
import XCTest
@testable import VoxtralCore

final class CancellationTests: XCTestCase {

    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    private func cLong() throws -> URL {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_CANCEL"] == "1",
                          "Set VOXTRAL_CANCEL=1 to run the cancellation tests")
        let url = root.appendingPathComponent(".local-runs/corpus/c_long.wav")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "C-long missing (PLAN.md §5)")
        return url
    }

    /// Starts `work`, cancels it after 3 s, returns the time from the cancellation to its end (ms)
    private func cancelAfterThreeSeconds(_ label: String, _ work: @escaping @Sendable () async throws -> Void) async throws -> Double {
        let task = Task { try await work() }
        try await Task.sleep(nanoseconds: 3_000_000_000)
        let cancelled = Date()
        task.cancel()
        do {
            try await task.value
            XCTFail("\(label): finished instead of being cancelled")
        } catch is CancellationError {
        } catch {
            XCTFail("\(label): expected CancellationError, got \(error)")
        }
        return Date().timeIntervalSince(cancelled) * 1000
    }

    func testCancelSTTOnCLong() async throws {
        let audio = try cLong()
        let pipeline = VoxtralPipeline(model: .mini3b8bit, backend: .mlx)
        try await pipeline.loadModel()
        let ms = try await cancelAfterThreeSeconds("STT") { _ = try await pipeline.transcribe(audio: audio) }
        print("[cancel] STT \(Int(ms)) ms, state \(pipeline.state)")
        XCTAssertLessThan(ms, 2000)
        XCTAssertTrue(pipeline.isReady)
    }

    /// Same on the default backend `.auto` (Core ML encoder): its window loop had no cancellation point (K-15, 2026-10-03)
    func testCancelSTTOnCLongAuto() async throws {
        let audio = try cLong()
        let pipeline = VoxtralPipeline(model: .mini3b8bit)
        try await pipeline.loadModel()
        let status = pipeline.encoderStatus
        XCTAssertTrue(status.contains("Core ML available: true"), "the test must run the Core ML encoder: \(status)")
        let ms = try await cancelAfterThreeSeconds("STT .auto") { _ = try await pipeline.transcribe(audio: audio) }
        print("[cancel] STT .auto \(Int(ms)) ms, state \(pipeline.state)")
        XCTAssertLessThan(ms, 2000)
        XCTAssertTrue(pipeline.isReady)
    }

    func testCancelTTSBatchOnLongText() async throws {
        _ = try cLong()
        let doc = try String(contentsOf: root.appendingPathComponent("docs/tts_benchmark.md"), encoding: .utf8)
        let lines = doc.components(separatedBy: "\n")
        let start = try XCTUnwrap(lines.firstIndex(of: "### Long EN"))
        let text = lines[(start + 1)...].prefix { !$0.hasPrefix("#") }.filter { $0.hasPrefix("> ") }
            .map { String($0.dropFirst(2)) }.joined(separator: " ")
        let pipeline = VoxtralTTSPipeline()
        try await pipeline.loadModel(modelInfo: try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-4bit")))
        let ms = try await cancelAfterThreeSeconds("TTS") {
            _ = try await pipeline.synthesize(text: text + " " + text, voice: .neutralFemale, seed: 42)
        }
        print("[cancel] TTS \(Int(ms)) ms, state \(pipeline.state)")
        XCTAssertLessThan(ms, 2000)
        XCTAssertTrue(pipeline.isReady)
    }

    func testCancelRealtimeOnCLong() async throws {
        let audio = try cLong()
        let pipeline = VoxtralRealtimePipeline()
        try await pipeline.loadModel(modelId: "realtime-4b-4bit")
        let ms = try await cancelAfterThreeSeconds("RT") { _ = try await pipeline.transcribe(audio: audio) }
        print("[cancel] RT \(Int(ms)) ms, state \(pipeline.state)")
        XCTAssertLessThan(ms, 2000)
        XCTAssertTrue(pipeline.isReady)
    }
}
