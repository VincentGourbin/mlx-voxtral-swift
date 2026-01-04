/**
 * VoxtralErrorTests - Unit tests for VoxtralError enum
 */

import XCTest
@testable import VoxtralCore

final class VoxtralErrorTests: XCTestCase {

    // MARK: - Error Creation Tests

    func testFileNotFoundError() {
        let error = VoxtralError.fileNotFound("/path/to/file.txt")

        if case .fileNotFound(let path) = error {
            XCTAssertEqual(path, "/path/to/file.txt")
        } else {
            XCTFail("Expected fileNotFound error")
        }
    }

    func testInvalidConfigurationError() {
        let error = VoxtralError.invalidConfiguration("Missing required field")

        if case .invalidConfiguration(let message) = error {
            XCTAssertEqual(message, "Missing required field")
        } else {
            XCTFail("Expected invalidConfiguration error")
        }
    }

    func testLoadingFailedError() {
        let error = VoxtralError.loadingFailed("Could not load weights")

        if case .loadingFailed(let message) = error {
            XCTAssertEqual(message, "Could not load weights")
        } else {
            XCTFail("Expected loadingFailed error")
        }
    }

    func testModelNotLoadedError() {
        let error = VoxtralError.modelNotLoaded

        if case .modelNotLoaded = error {
            // Success
        } else {
            XCTFail("Expected modelNotLoaded error")
        }
    }

    func testProcessorNotLoadedError() {
        let error = VoxtralError.processorNotLoaded

        if case .processorNotLoaded = error {
            // Success
        } else {
            XCTFail("Expected processorNotLoaded error")
        }
    }

    func testAudioProcessingFailedError() {
        let error = VoxtralError.audioProcessingFailed("Invalid audio format")

        if case .audioProcessingFailed(let message) = error {
            XCTAssertEqual(message, "Invalid audio format")
        } else {
            XCTFail("Expected audioProcessingFailed error")
        }
    }

    func testGenerationFailedError() {
        let error = VoxtralError.generationFailed("Token limit exceeded")

        if case .generationFailed(let message) = error {
            XCTAssertEqual(message, "Token limit exceeded")
        } else {
            XCTFail("Expected generationFailed error")
        }
    }

    func testTokenizerNotAvailableError() {
        let error = VoxtralError.tokenizerNotAvailable

        if case .tokenizerNotAvailable = error {
            // Success
        } else {
            XCTFail("Expected tokenizerNotAvailable error")
        }
    }

    func testInvalidTokenFormatError() {
        let error = VoxtralError.invalidTokenFormat

        if case .invalidTokenFormat = error {
            // Success
        } else {
            XCTFail("Expected invalidTokenFormat error")
        }
    }

    func testInvalidInputError() {
        let error = VoxtralError.invalidInput("Empty input provided")

        if case .invalidInput(let message) = error {
            XCTAssertEqual(message, "Empty input provided")
        } else {
            XCTFail("Expected invalidInput error")
        }
    }

    func testTokenizerRequiredError() {
        let error = VoxtralError.tokenizerRequired("Decoding operation")

        if case .tokenizerRequired(let operation) = error {
            XCTAssertEqual(operation, "Decoding operation")
        } else {
            XCTFail("Expected tokenizerRequired error")
        }
    }

    func testLanguageNotSupportedError() {
        let error = VoxtralError.languageNotSupported("xyz")

        if case .languageNotSupported(let language) = error {
            XCTAssertEqual(language, "xyz")
        } else {
            XCTFail("Expected languageNotSupported error")
        }
    }

    func testConfigurationNotFoundError() {
        let error = VoxtralError.configurationNotFound

        if case .configurationNotFound = error {
            // Success
        } else {
            XCTFail("Expected configurationNotFound error")
        }
    }

    // MARK: - Error Conformance Tests

    func testErrorConformance() {
        let errors: [Error] = [
            VoxtralError.fileNotFound("test"),
            VoxtralError.invalidConfiguration("test"),
            VoxtralError.loadingFailed("test"),
            VoxtralError.modelNotLoaded,
            VoxtralError.processorNotLoaded,
            VoxtralError.audioProcessingFailed("test"),
            VoxtralError.generationFailed("test"),
            VoxtralError.tokenizerNotAvailable,
            VoxtralError.invalidTokenFormat,
            VoxtralError.invalidInput("test"),
            VoxtralError.tokenizerRequired("test"),
            VoxtralError.languageNotSupported("test"),
            VoxtralError.configurationNotFound
        ]

        // All should conform to Error protocol
        XCTAssertEqual(errors.count, 13)

        for error in errors {
            XCTAssertNotNil(error.localizedDescription)
        }
    }

    // MARK: - Error Equality Tests

    func testErrorsWithSameValuesAreEqual() {
        let error1 = VoxtralError.fileNotFound("/path/test.txt")
        let error2 = VoxtralError.fileNotFound("/path/test.txt")

        // Using pattern matching since Error doesn't conform to Equatable
        if case .fileNotFound(let path1) = error1,
           case .fileNotFound(let path2) = error2 {
            XCTAssertEqual(path1, path2)
        } else {
            XCTFail("Expected both to be fileNotFound errors")
        }
    }

    func testErrorsWithDifferentValuesAreDifferent() {
        let error1 = VoxtralError.fileNotFound("/path/test1.txt")
        let error2 = VoxtralError.fileNotFound("/path/test2.txt")

        if case .fileNotFound(let path1) = error1,
           case .fileNotFound(let path2) = error2 {
            XCTAssertNotEqual(path1, path2)
        } else {
            XCTFail("Expected both to be fileNotFound errors")
        }
    }

    // MARK: - Error Message Content Tests

    func testFileNotFoundErrorContainsPath() {
        let path = "/very/specific/path/to/model.safetensors"
        let error = VoxtralError.fileNotFound(path)

        if case .fileNotFound(let errorPath) = error {
            XCTAssertTrue(errorPath.contains("model.safetensors"))
            XCTAssertTrue(errorPath.contains("/very/specific/path"))
        }
    }

    func testInvalidConfigurationErrorContainsDetails() {
        let details = "vocab_size must be positive"
        let error = VoxtralError.invalidConfiguration(details)

        if case .invalidConfiguration(let message) = error {
            XCTAssertTrue(message.contains("vocab_size"))
        }
    }

    // MARK: - Error Throwing Tests

    func testThrowingFileNotFoundError() {
        func throwingFunction() throws {
            throw VoxtralError.fileNotFound("/test/path")
        }

        XCTAssertThrowsError(try throwingFunction()) { error in
            guard case VoxtralError.fileNotFound(_) = error else {
                XCTFail("Expected fileNotFound error")
                return
            }
        }
    }

    func testThrowingModelNotLoadedError() {
        func throwingFunction() throws {
            throw VoxtralError.modelNotLoaded
        }

        XCTAssertThrowsError(try throwingFunction()) { error in
            guard case VoxtralError.modelNotLoaded = error else {
                XCTFail("Expected modelNotLoaded error")
                return
            }
        }
    }

    // MARK: - Edge Cases

    func testErrorWithEmptyString() {
        let error = VoxtralError.fileNotFound("")

        if case .fileNotFound(let path) = error {
            XCTAssertTrue(path.isEmpty)
        }
    }

    func testErrorWithSpecialCharacters() {
        let specialPath = "/path/with spaces/and émojis 🎵/file.txt"
        let error = VoxtralError.fileNotFound(specialPath)

        if case .fileNotFound(let path) = error {
            XCTAssertEqual(path, specialPath)
        }
    }

    func testErrorWithVeryLongMessage() {
        let longMessage = String(repeating: "x", count: 10000)
        let error = VoxtralError.invalidConfiguration(longMessage)

        if case .invalidConfiguration(let message) = error {
            XCTAssertEqual(message.count, 10000)
        }
    }
}
