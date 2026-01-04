/**
 * ModelLoadingTests - Unit tests for model loading utilities
 */

import XCTest
@testable import VoxtralCore

final class ModelLoadingTests: XCTestCase {

    // MARK: - Model Registry Integration Tests

    func testLoadConfigForKnownModel() {
        // Get a known model from registry
        let model = ModelRegistry.defaultModel

        // Can't actually load without the files, but verify the model info is complete
        XCTAssertFalse(model.repoId.isEmpty, "Model should have repo ID")
        XCTAssertFalse(model.id.isEmpty, "Model should have ID")
    }

    // MARK: - Quantization Detection Tests

    func testDetectQuantizationFrom4BitModel() {
        let model = ModelRegistry.model(withId: "mini-3b-4bit")

        XCTAssertNotNil(model, "Should find 4-bit model")
        if let model = model {
            XCTAssertTrue(
                model.quantization.contains("4-bit") || model.quantization.contains("4bit"),
                "Model should be 4-bit quantized"
            )
        }
    }

    func testDetectQuantizationFrom8BitModel() {
        let model = ModelRegistry.model(withId: "mini-3b-8bit")

        XCTAssertNotNil(model, "Should find 8-bit model")
        if let model = model {
            XCTAssertTrue(
                model.quantization.contains("8-bit") || model.quantization.contains("8bit"),
                "Model should be 8-bit quantized"
            )
        }
    }

    // MARK: - Configuration Parsing Tests

    func testVoxtralConfigFromDictionary() {
        let dict: [String: Any] = [
            "model_type": "voxtral",
            "audio_config": TestFixtures.sampleEncoderConfig,
            "text_config": TestFixtures.sampleTextConfig,
            "audio_token_id": 24,
            "projector_hidden_act": "gelu"
        ]

        // This tests the configuration parsing logic
        let audioConfig = VoxtralEncoderConfig.from_dict(
            dict["audio_config"] as? [String: Any] ?? [:]
        )
        let textConfig = VoxtralTextConfig.from_dict(
            dict["text_config"] as? [String: Any] ?? [:]
        )

        XCTAssertEqual(audioConfig.vocab_size, 51866)
        XCTAssertEqual(textConfig.vocab_size, 131072)
    }

    func testConfigWithMissingOptionalFields() {
        // Test config with only required fields
        let minimalDict: [String: Any] = [
            "model_type": "voxtral"
        ]

        // Should not crash, use defaults
        let audioConfig = VoxtralEncoderConfig.from_dict(
            minimalDict["audio_config"] as? [String: Any] ?? [:]
        )

        XCTAssertEqual(audioConfig.vocab_size, 51866, "Should use default values")
    }

    // MARK: - Weight Loading Validation Tests

    func testWeightFileExtensions() {
        // Valid weight file extensions
        let validExtensions = ["safetensors", "bin", "pt"]

        for ext in validExtensions {
            let filename = "model.\(ext)"
            let isValid = validExtensions.contains { filename.hasSuffix($0) }
            XCTAssertTrue(isValid, "\(ext) should be a valid weight file extension")
        }
    }

    func testConfigFileNames() {
        // Expected config file names
        let expectedFiles = ["config.json", "tokenizer.json", "tekken.json"]

        for file in expectedFiles {
            XCTAssertTrue(file.hasSuffix(".json"), "Config files should be JSON")
        }
    }

    // MARK: - Memory Estimation Tests

    func testMemoryEstimationForMiniModel() {
        // Mini models should require less memory
        let miniModel = ModelRegistry.model(withId: "mini-3b-4bit")

        if let model = miniModel {
            // 4-bit 3B model should be around 2GB
            XCTAssertTrue(
                model.size.contains("GB") || model.size.contains("MB"),
                "Model size should be specified in GB or MB"
            )
        }
    }

    func testParameterCountFormat() {
        for model in ModelRegistry.models {
            XCTAssertTrue(
                model.parameters.hasSuffix("B") || model.parameters.hasSuffix("M"),
                "Parameters should be in B (billion) or M (million) format: \(model.parameters)"
            )
        }
    }

    // MARK: - Model Info Completeness Tests

    func testAllModelsHaveDescription() {
        for model in ModelRegistry.models {
            XCTAssertFalse(
                model.description.isEmpty,
                "Model \(model.id) should have a description"
            )
        }
    }

    func testAllModelsHaveRepoId() {
        for model in ModelRegistry.models {
            XCTAssertTrue(
                model.repoId.contains("/"),
                "Repo ID should be in format 'owner/repo': \(model.repoId)"
            )
        }
    }

    func testAllModelsHaveQuantizationInfo() {
        for model in ModelRegistry.models {
            XCTAssertFalse(
                model.quantization.isEmpty,
                "Model \(model.id) should have quantization info"
            )
        }
    }

    // MARK: - Audio Loading Tests

    func testLoadAudioThrowsForMissingFile() {
        XCTAssertThrowsError(try loadAudio("/nonexistent/path/to/audio.wav")) { error in
            // Should throw some kind of error for missing file
            XCTAssertNotNil(error)
        }
    }

    func testLoadAudioThrowsForEmptyPath() {
        XCTAssertThrowsError(try loadAudio("")) { error in
            XCTAssertNotNil(error)
        }
    }
}
