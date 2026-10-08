/**
 * TTSStreamingFinalChunkTests - K-92 (constat de K-35)
 *
 * When EOA falls right after a chunk boundary, `generateStreaming` yields a final chunk with no new frame and the
 * same accumulated codes. `synthesizeStreaming` then took the "first slice" branch and emitted the whole content a
 * second time: the stream carried 2 × the audio (5 streaming cells of 29 in K-35, e.g. tts-4b-mlx, short_en, seed 3:
 * 73 frames, 11.68 s instead of 5.84 s).
 *
 * The heavy case loads tts-4b-mlx: skipped unless VOXTRAL_TTS_STREAM_FINAL=1
 * (`TEST_RUNNER_VOXTRAL_TTS_STREAM_FINAL=1 xcodebuild test …`).
 */

import Foundation
import MLX
import XCTest
@testable import VoxtralCore

final class TTSStreamingFinalChunkTests: XCTestCase {

    func testSliceStartsAfterWhatWasEmitted() {
        XCTAssertEqual(VoxtralTTSPipeline.newContentStart(previous: 0, contentTotal: 5_760), 0)
        XCTAssertEqual(VoxtralTTSPipeline.newContentStart(previous: 5_760, contentTotal: 19_200), 5_760)
    }

    func testChunkWithoutNewSampleEmitsNothing() {
        // Final chunk after EOA at a chunk boundary: same content as the previous chunk
        XCTAssertEqual(VoxtralTTSPipeline.newContentStart(previous: 140_160, contentTotal: 140_160), 140_160)
        XCTAssertEqual(VoxtralTTSPipeline.newContentStart(previous: 140_160, contentTotal: 138_240), 138_240)
    }

    func testStreamCarriesEachFrameOnce() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VOXTRAL_TTS_STREAM_FINAL"] == "1",
                          "Set VOXTRAL_TTS_STREAM_FINAL=1 to run the streaming final-chunk test")
        let pipeline = VoxtralTTSPipeline()
        let info = try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-mlx"))
        try await pipeline.loadModel(modelInfo: info)
        defer { pipeline.unload() }
        let textURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("docs/eval/tts/short_en.txt")
        let text = try String(contentsOf: textURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)

        // Same call as `bench tts --streaming --voice neutral_female --seed 3` (voice as an embedding)
        let voice = try XCTUnwrap(pipeline.voiceEmbeddings[VoxtralVoice.neutralFemale.rawValue])
        var samples = 0, frames = 0, finals = 0
        for try await chunk in pipeline.synthesizeStreaming(text: text, voiceEmbedding: voice, seed: 3, warmUpText: nil) {
            samples += chunk.waveform.dim(0)
            frames = chunk.totalFrames
            if chunk.isFinal { finals += 1 }
        }
        print("[stream-final] frames \(frames), samples \(samples) (\(String(format: "%.2f", Double(samples) / Double(max(frames, 1) * 1_920))) × frames)")
        XCTAssertEqual(finals, 1)
        XCTAssertEqual(samples, frames * 1_920)
    }
}
