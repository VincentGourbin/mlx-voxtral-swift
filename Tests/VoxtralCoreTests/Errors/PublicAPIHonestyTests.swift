/**
 * PublicAPIHonestyTests - K-27
 *
 * The public API does not lie and does not stop the process: `tokenCount` counts the generated tokens, and every
 * stop that a public call could reach (unsupported module type, missing input, too short a reference) is a typed
 * error at the throwing entry points. One test per case. The real-model cases are skipped when the model is absent.
 */

import Foundation
import MLX
import MLXNN
import XCTest
@testable import VoxtralCore

final class PublicAPIHonestyTests: XCTestCase {

    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    private let inputIds = MLXArray((0 ..< 16).map { Int32(30 + $0) }).reshaped([1, 16])

    private func assertInvalidConfiguration(_ expression: @autoclosure () throws -> some Any, _ hint: String,
                                            file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try expression(), file: file, line: line) { error in
            guard case VoxtralError.invalidConfiguration(let message) = error else {
                return XCTFail("expected VoxtralError.invalidConfiguration, got \(error)", file: file, line: line)
            }
            XCTAssertTrue(message.contains(hint), message, file: file, line: line)
        }
    }

    // tokenCount: the number of generated tokens, not 0
    @MainActor
    func testTokenCountOfATranscription() async throws {
        let info = try XCTUnwrap(VoxtralModelRegistry.model(withId: "mini-3b-8bit"))
        try XCTSkipIf(VoxtralModelDownloader.findModelPath(for: info) == nil, "mini-3b-8bit is not downloaded")
        let manager = VoxtralTranscriptionManager(model: .mini3b8bit)
        try await manager.loadModel()
        let result = try await manager.transcribe(
            audioURL: root.appendingPathComponent("docs/examples/fluxforge_short_en_6bit.wav"), language: "en")
        print("[honesty] tokenCount=\(result.tokenCount) for \"\(result.text)\"")
        XCTAssertGreaterThan(result.tokenCount, 0)
    }

    // STT decoder of an unsupported type (public `language_model: Module`)
    func testUnsupportedLanguageModelThrows() throws {
        let model = try makeReducedVoxtralModel()
        model.update(modules: ModuleChildren.unflattened(["language_model": Linear(4, 4)]))
        assertInvalidConfiguration(try model.generateStream(inputIds: inputIds, maxNewTokens: 1), "language_model")
    }

    // STT head of an unsupported type (public `lm_head: Module`)
    func testUnsupportedLMHeadThrows() throws {
        let model = try makeReducedVoxtralModel()
        model.update(modules: ModuleChildren.unflattened(["lm_head": RMSNorm(dimensions: 64)]))
        assertInvalidConfiguration(try model.generateStream(inputIds: inputIds, maxNewTokens: 1), "lm_head")
    }

    // Forward pass without inputIds nor inputsEmbeds, inside an MLX error boundary
    func testForwardWithoutInputsThrowsInsideABoundary() throws {
        let model = try makeReducedVoxtralModel()
        assertInvalidConfiguration(try withMLXErrors { errors in
            _ = model.callAsFunction(inputIds: nil, attentionMask: nil, inputFeatures: nil, inputsEmbeds: nil, pastKeyValues: nil)
            try errors.check()
        }, "input_ids")
    }

    // Layer count of an unsupported decoder, inside an MLX error boundary
    func testLayerCountOfAnUnsupportedDecoderThrowsInsideABoundary() throws {
        let model = try makeReducedVoxtralModel()
        model.update(modules: ModuleChildren.unflattened(["language_model": Linear(4, 4)]))
        assertInvalidConfiguration(try withMLXErrors { errors in
            XCTAssertEqual(model.getLanguageModelLayerCount(), 0)
            try errors.check()
        }, "language_model")
    }

    // TTS codebook embeddings of an unsupported type, inside an MLX error boundary
    func testUnsupportedCodebookEmbeddingsThrowInsideABoundary() throws {
        let container = AudioCodebookEmbeddingsContainer(totalSize: 10, dim: 4)
        container.update(modules: ModuleChildren.unflattened(["embeddings": Linear(4, 4)]))
        assertInvalidConfiguration(try withMLXErrors { errors in
            _ = container(MLXArray([Int32(1), 2]))
            try errors.check()
        }, "embeddings")
    }

    // TTS token embeddings of an unsupported type: the synthesis throws before any forward pass
    func testUnsupportedTTSTokenEmbeddingsThrowAtSynthesis() async throws {
        let info = try XCTUnwrap(VoxtralTTSRegistry.model(withId: "tts-4b-4bit"))
        try XCTSkipIf(VoxtralModelDownloader.findTTSModelPath(for: info) == nil, "tts-4b-4bit is not downloaded")
        let pipeline = VoxtralTTSPipeline()
        try await pipeline.loadModel(modelInfo: info)
        let model = try XCTUnwrap(pipeline.ttsModel)
        model.mmAudioEmbeddings.update(modules: ModuleChildren.unflattened(["tok_embeddings": Linear(4, 4)]))
        do {
            _ = try await pipeline.synthesize(text: "Hello.", voice: .neutralFemale, seed: 1)
            XCTFail("expected an error")
        } catch VoxtralTTSError.invalidConfiguration(let message) {
            XCTAssertTrue(message.contains("tok_embeddings"), message)
        }
        XCTAssertTrue(pipeline.isReady)
    }

    // Spectral losses on a reference shorter than the smallest FFT
    func testShortReferenceForLossesThrows() {
        XCTAssertThrowsError(try EnrollmentLossComputer(validating: MLXArray.zeros([40]))) { error in
            guard case VoxtralTTSError.invalidConfiguration = error else { return XCTFail("\(error)") }
        }
    }
}
