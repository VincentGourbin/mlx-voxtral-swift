/**
 * TTSStreamingCancellationTests - K-12 (S-08, MLX-003)
 *
 * Streaming TTS is really progressive and cancellable: `generateStreaming` returns its stream at once and
 * produces in a Task tied to `onTermination`; when the consumer stops, the producer stops within one frame and
 * the pipeline is `.ready` again. Before, the whole generation ran inside the stream's build closure.
 *
 * Long text (the "Long EN" text of docs/tts_benchmark.md, twice ≈ 330 words), tts-4b-4bit, no warm-up, seed 42.
 * Heavy (loads the real model): skipped unless VOXTRAL_TTS_STREAM=1 (`TEST_RUNNER_VOXTRAL_TTS_STREAM=1 xcodebuild test …`).
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class TTSStreamingCancellationTests: XCTestCase {

    nonisolated(unsafe) private static var shared: VoxtralTTSPipeline?

    private let seed: UInt64 = 42

    private func pipeline() async throws -> VoxtralTTSPipeline {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_TTS_STREAM"] == "1",
                          "Set VOXTRAL_TTS_STREAM=1 to run the streaming cancellation tests")
        if let shared = Self.shared { return shared }
        // Raw output on both paths, so that batch and concatenated stream are comparable
        var configuration = VoxtralTTSPipeline.Configuration.default
        configuration.trimLeadIn = false
        configuration.trimTail = false
        let pipeline = VoxtralTTSPipeline(configuration: configuration)
        let info = try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-4bit"))
        try await pipeline.loadModel(modelInfo: info)
        Self.shared = pipeline
        return pipeline
    }

    private func longText() throws -> String {
        let doc = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("docs/tts_benchmark.md")
        let lines = try String(contentsOf: doc, encoding: .utf8).components(separatedBy: "\n")
        let start = try XCTUnwrap(lines.firstIndex(of: "### Long EN"))
        let paragraphs = lines[(start + 1)...].prefix { !$0.hasPrefix("#") }
            .filter { $0.hasPrefix("> ") }.map { String($0.dropFirst(2)) }
        let text = paragraphs.joined(separator: " ")
        return text + " " + text
    }

    func testGenerateStreamingReturnsImmediately() async throws {
        let pipeline = try await pipeline()
        let model = try XCTUnwrap(pipeline.ttsModel), tokenizer = try XCTUnwrap(pipeline.tokenizer)
        let voice = try XCTUnwrap(pipeline.voiceEmbeddings[VoxtralVoice.neutralFemale.rawValue])
        let start = Date()
        let stream = model.generateStreaming(text: try longText(), voiceEmbedding: voice, tokenizer: tokenizer, seed: seed)
        let returnedMs = Date().timeIntervalSince(start) * 1000
        var iterator = stream.makeAsyncIterator()
        _ = try await iterator.next()  // the producer is alive; dropping the iterator stops it
        print("[stream] generateStreaming returned in \(String(format: "%.1f", returnedMs)) ms")
        XCTAssertLessThan(returnedMs, 50)
    }

    func testFirstChunkWithinBatchTTFT() async throws {
        let pipeline = try await pipeline()
        let text = try longText()
        let batch = try await pipeline.synthesize(text: text, voice: .neutralFemale, seed: seed)
        let ttftMs = batch.timeToFirstToken * 1000
        let start = Date()
        var firstMs = 0.0
        for try await chunk in pipeline.synthesizeStreaming(text: text, voice: .neutralFemale, seed: seed) where chunk.isFirst {
            firstMs = Date().timeIntervalSince(start) * 1000
            break
        }
        try await waitForReady(pipeline)
        print("[stream] first chunk \(String(format: "%.0f", firstMs)) ms ≤ 1,5 × ttft batch \(String(format: "%.0f", ttftMs)) ms")
        XCTAssertGreaterThan(firstMs, 0)
        // The gate (≤ 1,5 × ttft) is measured in Release by `bench tts` (a timed Debug test is not a reference
        // measurement, CLAUDE.md); here a guard: the first chunk no longer waits for the whole generation
        XCTAssertLessThanOrEqual(firstMs, 2 * ttftMs)
    }

    func testCancelAfterFiveChunks() async throws {
        let pipeline = try await pipeline()
        let model = try XCTUnwrap(pipeline.ttsModel)
        let text = try longText()
        let fiveChunks = expectation(description: "5 chunks received")
        nonisolated(unsafe) let consumerPipeline = pipeline
        let seed = self.seed
        let consumer = Task {
            var received = 0
            for try await _ in consumerPipeline.synthesizeStreaming(text: text, voice: .neutralFemale, seed: seed) {
                received += 1
                if received == 5 { fiveChunks.fulfill() }
            }
        }
        await fulfillment(of: [fiveChunks], timeout: 120)
        let framesAtCancel = model.streamingFramesProduced.get()
        let cancelled = Date()
        consumer.cancel()
        try await waitForReady(pipeline)
        let readyMs = Date().timeIntervalSince(cancelled) * 1000
        try await Task.sleep(nanoseconds: 500_000_000)  // a producer still running would keep counting
        let frames = model.streamingFramesProduced.get()
        print("[stream] CANCEL after 5 chunks → .ready in \(String(format: "%.0f", readyMs)) ms ; frames=\(frames) ≤ frames_at_cancel+1 (\(framesAtCancel + 1))")
        XCTAssertLessThan(readyMs, 1000)
        XCTAssertLessThanOrEqual(frames, framesAtCancel + 1)
    }

    func testStreamConcatEqualsBatch() async throws {
        let pipeline = try await pipeline()
        let text = try longText()
        // Generation: the streamed codes are the batch codes, bit for bit
        let model = try XCTUnwrap(pipeline.ttsModel), tokenizer = try XCTUnwrap(pipeline.tokenizer)
        let voice = try XCTUnwrap(pipeline.voiceEmbeddings[VoxtralVoice.neutralFemale.rawValue])
        let maxFrames = pipeline.configuration.maxFrames
        let batchCodes = model.generate(text: text, voiceEmbedding: voice, tokenizer: tokenizer,
                                        maxTokens: maxFrames, seed: seed).codes
        var streamCodes: MLXArray?
        for try await chunk in model.generateStreaming(text: text, voiceEmbedding: voice, tokenizer: tokenizer,
                                                       maxTokens: maxFrames, seed: seed) {
            streamCodes = chunk.accumulatedCodes
        }
        let codes = try XCTUnwrap(streamCodes)
        let codesEqual = codes.shape == batchCodes.shape && MLX.arrayEqual(codes, batchCodes).item(Bool.self)
        print("[stream] PARITY codes stream=\(codes.shape) batch=\(batchCodes.shape) identical=\(codesEqual)")
        XCTAssertTrue(codesEqual, "streamed codes == batch codes (seed 42)")

        // Audio: each chunk is cut from a re-decode of all codes so far (K-43)
        let batch = try await pipeline.synthesize(text: text, voice: .neutralFemale, seed: seed).waveform
        var parts: [MLXArray] = []
        for try await chunk in pipeline.synthesizeStreaming(text: text, voice: .neutralFemale, seed: seed)
        where chunk.waveform.dim(0) > 0 {
            parts.append(chunk.waveform)
        }
        let stream = MLX.concatenated(parts, axis: 0)
        let maxDiff = stream.dim(0) == batch.dim(0)
            ? MLX.abs(stream.asType(.float32) - batch.asType(.float32)).max().item(Float.self) : .infinity
        print("[stream] PARITY samples stream=\(stream.dim(0)) batch=\(batch.dim(0)) max|Δ|=\(maxDiff)")
        XCTAssertEqual(stream.dim(0), batch.dim(0))
        // Vincent, 2026-10-01: codes identical + audio max|Δ| ≤ 1e-5 (float noise of the per-chunk codec
        // re-decode; bit-identical audio belongs to K-43, incremental decoding)
        XCTAssertLessThanOrEqual(maxDiff, 1e-5, "stream concat == batch (seed 42)")
    }

    private func waitForReady(_ pipeline: VoxtralTTSPipeline, timeout: TimeInterval = 30) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !pipeline.state.isReady {
            guard Date() < deadline else { return XCTFail("pipeline not .ready after \(timeout) s: \(pipeline.state)") }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}
