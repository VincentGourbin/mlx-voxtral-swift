/**
 * VoxtralTTSRegistryTests - Unit tests for TTS model registry
 */

import XCTest
@testable import VoxtralCore

final class VoxtralTTSRegistryTests: XCTestCase {

    // MARK: - VoxtralTTSModelInfo Tests

    func testModelInfoProperties() {
        let info = VoxtralTTSModelInfo(
            id: "test-tts",
            repoId: "test/tts-model",
            name: "Test TTS Model",
            description: "A test TTS model",
            size: "8 GB",
            quantization: "bfloat16",
            parameters: "4B",
            recommended: true
        )

        XCTAssertEqual(info.id, "test-tts")
        XCTAssertEqual(info.repoId, "test/tts-model")
        XCTAssertEqual(info.name, "Test TTS Model")
        XCTAssertEqual(info.description, "A test TTS model")
        XCTAssertEqual(info.size, "8 GB")
        XCTAssertEqual(info.quantization, "bfloat16")
        XCTAssertEqual(info.parameters, "4B")
        XCTAssertTrue(info.recommended)
    }

    func testModelInfoDefaultRecommended() {
        let info = VoxtralTTSModelInfo(
            id: "test", repoId: "test/repo", name: "Test",
            description: "Desc", size: "1 GB", quantization: "fp16", parameters: "1B"
        )
        XCTAssertFalse(info.recommended)
    }

    func testModelInfoIdentifiable() {
        let info = VoxtralTTSModelInfo(
            id: "unique-id", repoId: "test/repo", name: "Test",
            description: "Desc", size: "1 GB", quantization: "fp16", parameters: "1B"
        )
        XCTAssertEqual(info.id, "unique-id")
    }

    // MARK: - Registry Models Tests

    func testRegistryModelsNotEmpty() {
        XCTAssertFalse(VoxtralTTSRegistry.models.isEmpty)
    }

    func testRegistryHasExpectedModelCount() {
        XCTAssertEqual(VoxtralTTSRegistry.models.count, 2)
    }

    func testAllModelsHaveRequiredFields() {
        for model in VoxtralTTSRegistry.models {
            XCTAssertFalse(model.id.isEmpty, "Model ID should not be empty")
            XCTAssertFalse(model.repoId.isEmpty, "Model repoId should not be empty")
            XCTAssertFalse(model.name.isEmpty, "Model name should not be empty")
            XCTAssertFalse(model.description.isEmpty, "Model description should not be empty")
            XCTAssertFalse(model.size.isEmpty, "Model size should not be empty")
            XCTAssertFalse(model.quantization.isEmpty, "Model quantization should not be empty")
            XCTAssertFalse(model.parameters.isEmpty, "Model parameters should not be empty")
        }
    }

    func testAllModelIdsAreUnique() {
        let ids = VoxtralTTSRegistry.models.map { $0.id }
        XCTAssertEqual(ids.count, Set(ids).count, "All model IDs should be unique")
    }

    func testAllRepoIdsAreUnique() {
        let repoIds = VoxtralTTSRegistry.models.map { $0.repoId }
        XCTAssertEqual(repoIds.count, Set(repoIds).count, "All repo IDs should be unique")
    }

    // MARK: - Model Lookup Tests

    func testModelLookupById() {
        let model = VoxtralTTSRegistry.model(withId: "tts-4b-mlx")
        XCTAssertNotNil(model)
        XCTAssertEqual(model?.id, "tts-4b-mlx")
    }

    func testModelLookupByIdOriginal() {
        let model = VoxtralTTSRegistry.model(withId: "tts-4b")
        XCTAssertNotNil(model)
        XCTAssertEqual(model?.id, "tts-4b")
    }

    func testModelLookupByIdNotFound() {
        let model = VoxtralTTSRegistry.model(withId: "non-existent")
        XCTAssertNil(model)
    }

    func testModelLookupIsCaseSensitive() {
        let model = VoxtralTTSRegistry.model(withId: "TTS-4B-MLX")
        XCTAssertNil(model, "Lookup should be case sensitive")
    }

    // MARK: - Default Model Tests

    func testDefaultModelExists() {
        let defaultModel = VoxtralTTSRegistry.defaultModel
        XCTAssertFalse(defaultModel.id.isEmpty)
    }

    func testDefaultModelIsRecommended() {
        let defaultModel = VoxtralTTSRegistry.defaultModel
        XCTAssertTrue(defaultModel.recommended)
    }

    func testDefaultModelIsMLXOptimized() {
        let defaultModel = VoxtralTTSRegistry.defaultModel
        XCTAssertEqual(defaultModel.id, "tts-4b-mlx")
        XCTAssertTrue(defaultModel.repoId.contains("mlx"))
    }

    func testAtLeastOneRecommendedModel() {
        let recommended = VoxtralTTSRegistry.models.filter { $0.recommended }
        XCTAssertFalse(recommended.isEmpty)
    }

    // MARK: - Model Content Tests

    func testMLXModelRepoId() {
        let model = VoxtralTTSRegistry.model(withId: "tts-4b-mlx")
        XCTAssertEqual(model?.repoId, "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16")
    }

    func testOriginalModelRepoId() {
        let model = VoxtralTTSRegistry.model(withId: "tts-4b")
        XCTAssertEqual(model?.repoId, "mistralai/Voxtral-4B-TTS-2603")
    }

    func testAllModelsAre4BParameters() {
        for model in VoxtralTTSRegistry.models {
            XCTAssertEqual(model.parameters, "4B")
        }
    }

    func testAllModelsAreBFloat16() {
        for model in VoxtralTTSRegistry.models {
            XCTAssertEqual(model.quantization, "bfloat16")
        }
    }
}
