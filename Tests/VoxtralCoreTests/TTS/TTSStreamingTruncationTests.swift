/**
 * TTSStreamingTruncationTests — K-86: `synthesizeStreaming` reports a frame cap reached through
 * `lastSynthesisTruncated`, like the batch paths (K-14); and the streaming generator, frame by frame (`chunkSize` 1),
 * yields the batch codes at a fixed seed (K-12 compares them by chunks of 10).
 *
 * Before K-86 only the batch paths set the flag: after a batch synthesis that ends on its own (false), a streaming
 * synthesis cut by its cap left it false.
 * Heavy (loads tts-4b-6bit): skipped unless VOXTRAL_TTS_STREAM=1 (`TEST_RUNNER_VOXTRAL_TTS_STREAM=1 xcodebuild test …`).
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class TTSStreamingTruncationTests: XCTestCase {

    nonisolated(unsafe) private static var shared: VoxtralTTSPipeline?

    private let text = "The quick brown fox jumps over the lazy dog, then it runs back home across the field."
    private let seed: UInt64 = 3

    private func pipeline() async throws -> VoxtralTTSPipeline {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_TTS_STREAM"] == "1",
                          "Set VOXTRAL_TTS_STREAM=1 to run the streaming truncation tests")
        if let shared = Self.shared { return shared }
        let pipeline = VoxtralTTSPipeline()
        try await pipeline.loadModel(modelInfo: try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-6bit")))
        Self.shared = pipeline
        return pipeline
    }

    private func drain(_ stream: AsyncThrowingStream<TTSStreamingChunk, Error>) async throws -> (frames: Int, final: Bool) {
        var frames = 0, final = false
        for try await chunk in stream {
            frames = chunk.totalFrames
            final = final || chunk.isFinal
        }
        return (frames, final)
    }

    func testStreamingCutByTheCapIsReportedTruncated() async throws {
        let pipeline = try await pipeline()
        let saved = pipeline.configuration
        defer { pipeline.configuration = saved }

        _ = try await pipeline.synthesize(text: text, voice: .neutralFemale, seed: seed)
        XCTAssertFalse(pipeline.lastSynthesisTruncated, "batch synthesis ends on its own")

        pipeline.configuration.maxFrames = 12
        let cut = try await drain(pipeline.synthesizeStreaming(text: text, voice: .neutralFemale, seed: seed))
        print("[k86] streaming with cap 12: \(cut.frames) frames, truncated=\(pipeline.lastSynthesisTruncated)")
        XCTAssertEqual(cut.frames, 12)
        XCTAssertTrue(cut.final)
        XCTAssertTrue(pipeline.lastSynthesisTruncated, "the cap was reached in streaming")

        pipeline.configuration = saved
        let whole = try await drain(pipeline.synthesizeStreaming(text: text, voice: .neutralFemale, seed: seed))
        print("[k86] streaming without cap: \(whole.frames) frames, truncated=\(pipeline.lastSynthesisTruncated)")
        XCTAssertLessThan(whole.frames, pipeline.frameCap(forText: text))
        XCTAssertFalse(pipeline.lastSynthesisTruncated, "a streaming synthesis that ends on its own is not truncated")
    }

    func testFrameByFrameStreamYieldsTheBatchCodes() async throws {
        let pipeline = try await pipeline()
        let model = try XCTUnwrap(pipeline.ttsModel), tokenizer = try XCTUnwrap(pipeline.tokenizer)
        let voice = try XCTUnwrap(pipeline.voiceEmbeddings[VoxtralVoice.neutralFemale.rawValue])
        let cap = pipeline.frameCap(forText: text)
        let batch = model.generate(text: text, voiceEmbedding: voice, tokenizer: tokenizer, maxTokens: cap, seed: seed).codes
        var last: MLXArray?
        var chunks = 0, previous = 0
        for try await chunk in model.generateStreaming(text: text, voiceEmbedding: voice, tokenizer: tokenizer,
                                                       maxTokens: cap, chunkSize: 1, seed: seed) {
            XCTAssertEqual(chunk.totalFrames, previous + chunk.newFrameCount)
            XCTAssertLessThanOrEqual(chunk.newFrameCount, 1, "one frame per chunk")
            XCTAssertEqual(chunk.accumulatedCodes.dim(1), chunk.totalFrames)
            previous = chunk.totalFrames
            last = chunk.accumulatedCodes
            chunks += 1
        }
        let codes = try XCTUnwrap(last)
        let equal = codes.shape == batch.shape && MLX.arrayEqual(codes, batch).item(Bool.self)
        print("[k86] PARITY frame-by-frame stream=\(codes.shape) batch=\(batch.shape) chunks=\(chunks) identical=\(equal)")
        XCTAssertGreaterThan(chunks, 10)
        XCTAssertTrue(equal, "frame-by-frame streamed codes == batch codes (seed \(seed))")
    }
}
