/**
 * ModelRegistryTests - Unit tests for ModelRegistry
 */

import XCTest
@testable import VoxtralCore

final class ModelRegistryTests: XCTestCase {

    // MARK: - VoxtralModelInfo Tests

    func testModelInfoCodableRoundTrip() throws {
        let original = VoxtralModelInfo(
            id: "test-model",
            repoId: "test/test-model",
            name: "Test Model",
            description: "A test model",
            size: "1 GB",
            quantization: "8-bit",
            parameters: "1B",
            recommended: true
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let restored = try decoder.decode(VoxtralModelInfo.self, from: data)

        XCTAssertEqual(original.id, restored.id)
        XCTAssertEqual(original.repoId, restored.repoId)
        XCTAssertEqual(original.name, restored.name)
        XCTAssertEqual(original.description, restored.description)
        XCTAssertEqual(original.size, restored.size)
        XCTAssertEqual(original.quantization, restored.quantization)
        XCTAssertEqual(original.parameters, restored.parameters)
        XCTAssertEqual(original.recommended, restored.recommended)
    }

    func testModelInfoIdentifiable() {
        let model = VoxtralModelInfo(
            id: "unique-id",
            repoId: "test/repo",
            name: "Test",
            description: "Description",
            size: "1 GB",
            quantization: "4-bit",
            parameters: "1B"
        )

        XCTAssertEqual(model.id, "unique-id")
    }

    func testModelInfoDefaultRecommended() {
        let model = VoxtralModelInfo(
            id: "test",
            repoId: "test/repo",
            name: "Test",
            description: "Desc",
            size: "1 GB",
            quantization: "4-bit",
            parameters: "1B"
        )

        XCTAssertFalse(model.recommended)
    }

    // MARK: - ModelRegistry.models Tests

    func testModelsNotEmpty() {
        XCTAssertFalse(ModelRegistry.models.isEmpty)
    }

    func testAllModelsHaveRequiredFields() {
        for model in ModelRegistry.models {
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
        let ids = ModelRegistry.models.map { $0.id }
        let uniqueIds = Set(ids)

        XCTAssertEqual(ids.count, uniqueIds.count, "All model IDs should be unique")
    }

    func testAllRepoIdsAreUnique() {
        let repoIds = ModelRegistry.models.map { $0.repoId }
        let uniqueRepoIds = Set(repoIds)

        XCTAssertEqual(repoIds.count, uniqueRepoIds.count, "All repo IDs should be unique")
    }

    // MARK: - ModelRegistry Lookup Tests

    func testModelLookupById() {
        // Test with known model IDs
        let model = ModelRegistry.model(withId: "mini-3b-8bit")

        XCTAssertNotNil(model)
        XCTAssertEqual(model?.id, "mini-3b-8bit")
        XCTAssertTrue(model?.repoId.contains("voxtral") ?? false)
    }

    func testModelLookupByIdNotFound() {
        let model = ModelRegistry.model(withId: "non-existent-model")

        XCTAssertNil(model)
    }

    func testModelLookupByRepoId() {
        let model = ModelRegistry.model(withRepoId: "mzbac/voxtral-mini-3b-8bit")

        XCTAssertNotNil(model)
        XCTAssertEqual(model?.repoId, "mzbac/voxtral-mini-3b-8bit")
    }

    func testModelLookupByRepoIdNotFound() {
        let model = ModelRegistry.model(withRepoId: "fake/non-existent-repo")

        XCTAssertNil(model)
    }

    // MARK: - Default Model Tests

    func testDefaultModelExists() {
        let defaultModel = ModelRegistry.defaultModel

        XCTAssertFalse(defaultModel.id.isEmpty)
        XCTAssertFalse(defaultModel.name.isEmpty)
    }

    func testDefaultModelIsRecommended() {
        let defaultModel = ModelRegistry.defaultModel

        XCTAssertTrue(defaultModel.recommended, "Default model should be marked as recommended")
    }

    func testAtLeastOneRecommendedModel() {
        let recommendedModels = ModelRegistry.models.filter { $0.recommended }

        XCTAssertFalse(recommendedModels.isEmpty, "There should be at least one recommended model")
    }

    // MARK: - Model Filtering Tests

    func testOfficialModelsFilter() {
        let officialModels = ModelRegistry.officialModels

        for model in officialModels {
            XCTAssertTrue(
                model.repoId.hasPrefix("mistralai/"),
                "Official model should have mistralai/ prefix: \(model.repoId)"
            )
        }
    }

    func testMiniModelsFilter() {
        let miniModels = ModelRegistry.miniModels

        for model in miniModels {
            XCTAssertEqual(model.parameters, "3B", "Mini model should have 3B parameters")
            XCTAssertFalse(
                model.repoId.hasPrefix("mistralai/"),
                "Mini models should not include official Mistral models"
            )
        }
    }

    func testSmallModelsFilter() {
        let smallModels = ModelRegistry.smallModels

        for model in smallModels {
            XCTAssertEqual(model.parameters, "24B", "Small model should have 24B parameters")
            XCTAssertFalse(
                model.repoId.hasPrefix("mistralai/"),
                "Small models should not include official Mistral models"
            )
        }
    }

    func testFiltersCoverAllModels() {
        let official = ModelRegistry.officialModels
        let mini = ModelRegistry.miniModels
        let small = ModelRegistry.smallModels

        // All models should be in one of the categories
        let totalFiltered = official.count + mini.count + small.count
        let allModels = ModelRegistry.models.count

        XCTAssertEqual(
            totalFiltered,
            allModels,
            "Sum of filtered models should equal total models"
        )
    }

    // MARK: - Model Size Categories

    func testMiniModelsHave3BParameters() {
        let miniModels = ModelRegistry.miniModels
        XCTAssertFalse(miniModels.isEmpty, "Should have mini models")

        for model in miniModels {
            XCTAssertEqual(model.parameters, "3B")
        }
    }

    func testSmallModelsHave24BParameters() {
        let smallModels = ModelRegistry.smallModels
        XCTAssertFalse(smallModels.isEmpty, "Should have small models")

        for model in smallModels {
            XCTAssertEqual(model.parameters, "24B")
        }
    }

    // MARK: - Quantization Types

    func testQuantizationTypesExist() {
        let quantizations = Set(ModelRegistry.models.map { $0.quantization })

        // Should have at least 4-bit and 8-bit quantization options
        XCTAssertTrue(
            quantizations.contains { $0.contains("4-bit") } ||
            quantizations.contains { $0.contains("float") },
            "Should have varied quantization options"
        )
    }

    // MARK: - Case Sensitivity Tests

    func testModelLookupIsCaseSensitive() {
        let modelLower = ModelRegistry.model(withId: "mini-3b-8bit")
        let modelUpper = ModelRegistry.model(withId: "MINI-3B-8BIT")

        XCTAssertNotNil(modelLower)
        XCTAssertNil(modelUpper, "Lookup should be case sensitive")
    }
}
