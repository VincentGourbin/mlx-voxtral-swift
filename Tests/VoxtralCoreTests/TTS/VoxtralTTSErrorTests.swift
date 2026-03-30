/**
 * VoxtralTTSErrorTests - Unit tests for VoxtralTTSError enum
 */

import XCTest
@testable import VoxtralCore

final class VoxtralTTSErrorTests: XCTestCase {

    // MARK: - Error Creation Tests

    func testFileNotFoundError() {
        let error = VoxtralTTSError.fileNotFound("params.json")
        if case .fileNotFound(let msg) = error {
            XCTAssertEqual(msg, "params.json")
        } else {
            XCTFail("Expected fileNotFound error")
        }
    }

    func testInvalidConfigurationError() {
        let error = VoxtralTTSError.invalidConfiguration("Missing dim field")
        if case .invalidConfiguration(let msg) = error {
            XCTAssertEqual(msg, "Missing dim field")
        } else {
            XCTFail("Expected invalidConfiguration error")
        }
    }

    func testModelLoadingFailedError() {
        let error = VoxtralTTSError.modelLoadingFailed("Weight mismatch")
        if case .modelLoadingFailed(let msg) = error {
            XCTAssertEqual(msg, "Weight mismatch")
        } else {
            XCTFail("Expected modelLoadingFailed error")
        }
    }

    func testSynthesisError() {
        let error = VoxtralTTSError.synthesisError("No frames generated")
        if case .synthesisError(let msg) = error {
            XCTAssertEqual(msg, "No frames generated")
        } else {
            XCTFail("Expected synthesisError error")
        }
    }

    func testVoiceNotFoundError() {
        let error = VoxtralTTSError.voiceNotFound("unknown_voice")
        if case .voiceNotFound(let msg) = error {
            XCTAssertEqual(msg, "unknown_voice")
        } else {
            XCTFail("Expected voiceNotFound error")
        }
    }

    // MARK: - LocalizedError Conformance

    func testFileNotFoundErrorDescription() {
        let error = VoxtralTTSError.fileNotFound("params.json")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("params.json"))
        XCTAssertTrue(error.errorDescription!.contains("File not found"))
    }

    func testInvalidConfigurationErrorDescription() {
        let error = VoxtralTTSError.invalidConfiguration("bad config")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("bad config"))
        XCTAssertTrue(error.errorDescription!.contains("Invalid configuration"))
    }

    func testModelLoadingFailedErrorDescription() {
        let error = VoxtralTTSError.modelLoadingFailed("weights missing")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("weights missing"))
        XCTAssertTrue(error.errorDescription!.contains("Model loading failed"))
    }

    func testSynthesisErrorDescription() {
        let error = VoxtralTTSError.synthesisError("timeout")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("timeout"))
        XCTAssertTrue(error.errorDescription!.contains("Synthesis error"))
    }

    func testVoiceNotFoundErrorDescription() {
        let error = VoxtralTTSError.voiceNotFound("ghost_voice")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("ghost_voice"))
        XCTAssertTrue(error.errorDescription!.contains("Voice not found"))
    }

    // MARK: - Error Conformance Tests

    func testAllErrorsConformToError() {
        let errors: [Error] = [
            VoxtralTTSError.fileNotFound("test"),
            VoxtralTTSError.invalidConfiguration("test"),
            VoxtralTTSError.modelLoadingFailed("test"),
            VoxtralTTSError.synthesisError("test"),
            VoxtralTTSError.voiceNotFound("test"),
        ]

        XCTAssertEqual(errors.count, 5)
        for error in errors {
            XCTAssertNotNil(error.localizedDescription)
            XCTAssertFalse(error.localizedDescription.isEmpty)
        }
    }

    // MARK: - Error Throwing Tests

    func testThrowingFileNotFoundError() {
        XCTAssertThrowsError(try throwTTSError(.fileNotFound("test"))) { error in
            guard case VoxtralTTSError.fileNotFound = error else {
                XCTFail("Expected fileNotFound"); return
            }
        }
    }

    func testThrowingVoiceNotFoundError() {
        XCTAssertThrowsError(try throwTTSError(.voiceNotFound("test"))) { error in
            guard case VoxtralTTSError.voiceNotFound = error else {
                XCTFail("Expected voiceNotFound"); return
            }
        }
    }

    func testThrowingSynthesisError() {
        XCTAssertThrowsError(try throwTTSError(.synthesisError("test"))) { error in
            guard case VoxtralTTSError.synthesisError = error else {
                XCTFail("Expected synthesisError"); return
            }
        }
    }

    // MARK: - Edge Cases

    func testErrorWithEmptyMessage() {
        let error = VoxtralTTSError.fileNotFound("")
        if case .fileNotFound(let msg) = error {
            XCTAssertTrue(msg.isEmpty)
        }
        XCTAssertNotNil(error.errorDescription)
    }

    func testErrorWithLongMessage() {
        let longMsg = String(repeating: "x", count: 10000)
        let error = VoxtralTTSError.synthesisError(longMsg)
        if case .synthesisError(let msg) = error {
            XCTAssertEqual(msg.count, 10000)
        }
    }

    func testErrorWithSpecialCharacters() {
        let specialMsg = "Error: café résumé 🎵 日本語"
        let error = VoxtralTTSError.invalidConfiguration(specialMsg)
        if case .invalidConfiguration(let msg) = error {
            XCTAssertEqual(msg, specialMsg)
        }
    }

    // MARK: - Helpers

    private func throwTTSError(_ error: VoxtralTTSError) throws {
        throw error
    }
}
