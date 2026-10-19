/**
 * TTSStreamingPrefixBoundaryTests — K-50 (K-1): the voice prefix is computed inside the MLX error boundary (in streaming,
 * inside the producing Task) and a failed prefix is not cached. An MLX error while prefilling the voice (here an embedding of the wrong width) ends the stream
 * with `VoxtralError.mlx`; computed on the caller's thread, before the Task, it called mlx-swift's default handler
 * (`fatalError`) and stopped the host (piège 38: without the fix this test crashes the runner).
 * Heavy (loads tts-4b-6bit): skipped unless VOXTRAL_TTS_STREAM=1 (`TEST_RUNNER_VOXTRAL_TTS_STREAM=1 xcodebuild test …`).
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class TTSStreamingPrefixBoundaryTests: XCTestCase {

    func testVoicePrefixMLXErrorEndsTheStream() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_TTS_STREAM"] == "1",
                          "Set VOXTRAL_TTS_STREAM=1 to run the streaming prefix boundary test")
        let pipeline = VoxtralTTSPipeline()
        try await pipeline.loadModel(modelInfo: try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-6bit")))
        defer { pipeline.unload() }

        // 100 columns instead of 3072: the voice frames cannot join the prompt embeddings
        let wrongWidth = MLXArray.zeros([5, 100]).asType(.bfloat16)
        do {
            for try await _ in pipeline.synthesizeStreaming(text: "Hello there.", voiceEmbedding: wrongWidth, seed: 1) {}
            XCTFail("expected the stream to end with an error")
        } catch {
            guard case VoxtralError.mlx = error else { return XCTFail("expected VoxtralError.mlx, got \(error)") }
            print("[k50] wrong-width voice embedding in streaming: \(error)")
        }
        XCTAssertTrue(pipeline.prefixCaches.entries.isEmpty, "no entry for a prefix that failed")

        // Same embedding through the batch overload (its prefix is cached since K-50)
        do {
            _ = try await pipeline.synthesize(text: "Hello there.", voiceEmbedding: wrongWidth, seed: 1)
            XCTFail("expected an error")
        } catch {
            guard case VoxtralError.mlx = error else { return XCTFail("expected VoxtralError.mlx, got \(error)") }
        }
        XCTAssertTrue(pipeline.prefixCaches.entries.isEmpty)

        // The pipeline stays usable
        let ok = try await pipeline.synthesize(text: "Hello there.", voice: .neutralFemale, seed: 1)
        XCTAssertGreaterThan(ok.waveform.size, 0)
    }
}
