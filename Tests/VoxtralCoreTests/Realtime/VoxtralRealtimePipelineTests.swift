/**
 * VoxtralRealtimePipelineTests - Tests for Realtime pipeline state machine and config
 */

import XCTest
@testable import VoxtralCore

@available(macOS 14.0, *)
final class VoxtralRealtimePipelineTests: XCTestCase {

    // MARK: - Configuration Tests

    func testDefaultConfiguration() {
        let config = VoxtralRealtimePipeline.Configuration.default
        XCTAssertEqual(config.maxTokens, 4096)
        XCTAssertEqual(config.temperature, 0.0)
        XCTAssertEqual(config.transcriptionDelayMs, 480)
    }

    func testCustomConfiguration() {
        let config = VoxtralRealtimePipeline.Configuration(
            maxTokens: 1000, temperature: 0.5, transcriptionDelayMs: 240
        )
        XCTAssertEqual(config.maxTokens, 1000)
        XCTAssertEqual(config.temperature, 0.5)
        XCTAssertEqual(config.transcriptionDelayMs, 240)
    }

    // MARK: - State Tests

    func testInitialState() {
        let pipeline = VoxtralRealtimePipeline()
        XCTAssertFalse(pipeline.isReady)
        if case .unloaded = pipeline.state { } else { XCTFail("Expected unloaded") }
    }

    func testSampleRate() {
        let pipeline = VoxtralRealtimePipeline()
        XCTAssertEqual(pipeline.sampleRate, 16000)
    }

    func testUnload() {
        let pipeline = VoxtralRealtimePipeline()
        pipeline.unload()
        XCTAssertFalse(pipeline.isReady)
    }

    // MARK: - Error on Unloaded

    func testTranscribeWithoutLoading() async {
        let pipeline = VoxtralRealtimePipeline()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("test.wav")
        do {
            _ = try await pipeline.transcribe(audio: url)
            XCTFail("Should throw")
        } catch {
            guard case VoxtralRealtimeError.invalidConfiguration = error else {
                XCTFail("Expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testExtractEmbeddingsWithoutLoading() async {
        let pipeline = VoxtralRealtimePipeline()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("test.wav")
        do {
            _ = try await pipeline.extractAudioEmbeddings(audio: url)
            XCTFail("Should throw")
        } catch {
            guard case VoxtralRealtimeError.invalidConfiguration = error else {
                XCTFail("Expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    // MARK: - Error Types

    func testRealtimeErrors() {
        let errors: [VoxtralRealtimeError] = [
            .fileNotFound("test"),
            .invalidConfiguration("test"),
            .modelLoadingFailed("test"),
            .transcriptionError("test"),
        ]
        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }
}
