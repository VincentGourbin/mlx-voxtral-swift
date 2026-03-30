/**
 * VoxtralTTSPipelineTests - Unit tests for TTS pipeline configuration and state machine
 */

import XCTest
@testable import VoxtralCore

@available(macOS 14.0, *)
final class VoxtralTTSPipelineTests: XCTestCase {

    // MARK: - Pipeline Configuration Tests

    func testDefaultConfiguration() {
        let config = VoxtralTTSPipeline.Configuration.default

        XCTAssertEqual(config.maxFrames, 2500)
        XCTAssertEqual(config.temperature, 0.0)
        XCTAssertEqual(config.cfgAlpha, 1.2)
        XCTAssertEqual(config.flowSteps, 8)
    }

    func testCustomConfiguration() {
        let config = VoxtralTTSPipeline.Configuration(
            maxFrames: 1000,
            temperature: 0.5,
            cfgAlpha: 1.5,
            flowSteps: 16
        )

        XCTAssertEqual(config.maxFrames, 1000)
        XCTAssertEqual(config.temperature, 0.5)
        XCTAssertEqual(config.cfgAlpha, 1.5)
        XCTAssertEqual(config.flowSteps, 16)
    }

    func testPartialCustomConfiguration() {
        let config = VoxtralTTSPipeline.Configuration(maxFrames: 500)

        XCTAssertEqual(config.maxFrames, 500)
        XCTAssertEqual(config.temperature, 0.0)  // default
        XCTAssertEqual(config.cfgAlpha, 1.2)     // default
        XCTAssertEqual(config.flowSteps, 8)      // default
    }

    // MARK: - Pipeline State Tests

    func testInitialStateIsUnloaded() {
        let pipeline = VoxtralTTSPipeline()

        XCTAssertFalse(pipeline.isReady)
        if case .unloaded = pipeline.state {
            // Expected
        } else {
            XCTFail("Expected unloaded state")
        }
    }

    func testStateIsUnloadedProperty() {
        let state = VoxtralTTSPipeline.State.unloaded
        XCTAssertTrue(state.isUnloaded)
        XCTAssertFalse(state.isReady)
    }

    func testStateIsReadyProperty() {
        let state = VoxtralTTSPipeline.State.ready
        XCTAssertTrue(state.isReady)
        XCTAssertFalse(state.isUnloaded)
    }

    func testStateLoadingIsNeitherReadyNorUnloaded() {
        let state = VoxtralTTSPipeline.State.loading
        XCTAssertFalse(state.isReady)
        XCTAssertFalse(state.isUnloaded)
    }

    func testStateErrorIsNeitherReadyNorUnloaded() {
        let state = VoxtralTTSPipeline.State.error("some error")
        XCTAssertFalse(state.isReady)
        XCTAssertFalse(state.isUnloaded)
    }

    // MARK: - Pipeline Initialization Tests

    func testPipelineDefaultInit() {
        let pipeline = VoxtralTTSPipeline()
        XCTAssertEqual(pipeline.sampleRate, 24000)
        XCTAssertFalse(pipeline.isReady)
        XCTAssertTrue(pipeline.availableVoices.isEmpty)
    }

    func testPipelineWithCustomConfig() {
        let config = VoxtralTTSPipeline.Configuration(maxFrames: 1000)
        let pipeline = VoxtralTTSPipeline(configuration: config)

        XCTAssertEqual(pipeline.configuration.maxFrames, 1000)
        XCTAssertEqual(pipeline.sampleRate, 24000)
    }

    func testPipelineConfigurationIsMutable() {
        let pipeline = VoxtralTTSPipeline()
        pipeline.configuration.maxFrames = 500
        XCTAssertEqual(pipeline.configuration.maxFrames, 500)
    }

    // MARK: - Pipeline Unload Tests

    func testUnloadResetsState() {
        let pipeline = VoxtralTTSPipeline()
        pipeline.unload()

        XCTAssertFalse(pipeline.isReady)
        XCTAssertTrue(pipeline.availableVoices.isEmpty)
    }

    // MARK: - Pipeline Synthesis Without Loading

    func testSynthesizeWithoutLoadingThrows() async {
        let pipeline = VoxtralTTSPipeline()

        do {
            _ = try await pipeline.synthesize(text: "Hello")
            XCTFail("Should throw when not loaded")
        } catch {
            guard case VoxtralTTSError.invalidConfiguration = error else {
                XCTFail("Expected invalidConfiguration error, got \(error)")
                return
            }
        }
    }

    func testSynthesizeToFileWithoutLoadingThrows() async {
        let pipeline = VoxtralTTSPipeline()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("test.wav")

        do {
            _ = try await pipeline.synthesizeToFile(text: "Hello", outputURL: url)
            XCTFail("Should throw when not loaded")
        } catch {
            // Expected
        }
    }

    // MARK: - VoxtralTTSSynthesisManager Tests

    func testManagerDefaultVoice() {
        let manager = VoxtralTTSSynthesisManager()
        XCTAssertEqual(manager.defaultVoice, .neutralFemale)
    }

    func testManagerCustomVoice() {
        let manager = VoxtralTTSSynthesisManager(voice: .frFemale)
        XCTAssertEqual(manager.defaultVoice, .frFemale)
    }

    func testManagerIsNotLoadedByDefault() {
        let manager = VoxtralTTSSynthesisManager()
        XCTAssertFalse(manager.isLoaded)
    }

    func testManagerStatusSummaryNotLoaded() {
        let manager = VoxtralTTSSynthesisManager()
        XCTAssertEqual(manager.statusSummary, "TTS model not loaded")
    }

    func testManagerAvailableVoicesWhenNotLoaded() {
        let manager = VoxtralTTSSynthesisManager()
        XCTAssertTrue(manager.availableVoices.isEmpty)
    }

    func testManagerUnload() {
        let manager = VoxtralTTSSynthesisManager()
        manager.unloadModel()  // Should not crash even when not loaded
        XCTAssertFalse(manager.isLoaded)
    }

    func testManagerDefaultVoiceIsMutable() {
        let manager = VoxtralTTSSynthesisManager()
        manager.defaultVoice = .casualMale
        XCTAssertEqual(manager.defaultVoice, .casualMale)
    }

    func testManagerWithCustomConfiguration() {
        let config = VoxtralTTSPipeline.Configuration(maxFrames: 1000, flowSteps: 4)
        let manager = VoxtralTTSSynthesisManager(voice: .deMale, configuration: config)

        XCTAssertEqual(manager.defaultVoice, .deMale)
        XCTAssertFalse(manager.isLoaded)
    }
}
