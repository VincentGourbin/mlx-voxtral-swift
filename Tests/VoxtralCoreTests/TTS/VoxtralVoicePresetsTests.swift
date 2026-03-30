/**
 * VoxtralVoicePresetsTests - Unit tests for voice presets and voice manager
 */

import XCTest
@testable import VoxtralCore

final class VoxtralVoicePresetsTests: XCTestCase {

    // MARK: - VoxtralVoice Enum Tests

    func testAllVoicesCaseCount() {
        XCTAssertEqual(VoxtralVoice.allCases.count, 20)
    }

    func testVoiceRawValues() {
        XCTAssertEqual(VoxtralVoice.casualFemale.rawValue, "casual_female")
        XCTAssertEqual(VoxtralVoice.casualMale.rawValue, "casual_male")
        XCTAssertEqual(VoxtralVoice.cheerfulFemale.rawValue, "cheerful_female")
        XCTAssertEqual(VoxtralVoice.neutralFemale.rawValue, "neutral_female")
        XCTAssertEqual(VoxtralVoice.neutralMale.rawValue, "neutral_male")
        XCTAssertEqual(VoxtralVoice.frMale.rawValue, "fr_male")
        XCTAssertEqual(VoxtralVoice.frFemale.rawValue, "fr_female")
        XCTAssertEqual(VoxtralVoice.esMale.rawValue, "es_male")
        XCTAssertEqual(VoxtralVoice.esFemale.rawValue, "es_female")
        XCTAssertEqual(VoxtralVoice.deMale.rawValue, "de_male")
        XCTAssertEqual(VoxtralVoice.deFemale.rawValue, "de_female")
        XCTAssertEqual(VoxtralVoice.itMale.rawValue, "it_male")
        XCTAssertEqual(VoxtralVoice.itFemale.rawValue, "it_female")
        XCTAssertEqual(VoxtralVoice.ptMale.rawValue, "pt_male")
        XCTAssertEqual(VoxtralVoice.ptFemale.rawValue, "pt_female")
        XCTAssertEqual(VoxtralVoice.nlMale.rawValue, "nl_male")
        XCTAssertEqual(VoxtralVoice.nlFemale.rawValue, "nl_female")
        XCTAssertEqual(VoxtralVoice.arMale.rawValue, "ar_male")
        XCTAssertEqual(VoxtralVoice.hiMale.rawValue, "hi_male")
        XCTAssertEqual(VoxtralVoice.hiFemale.rawValue, "hi_female")
    }

    func testAllRawValuesAreUnique() {
        let rawValues = VoxtralVoice.allCases.map { $0.rawValue }
        XCTAssertEqual(rawValues.count, Set(rawValues).count)
    }

    // MARK: - Display Name Tests

    func testDisplayNames() {
        XCTAssertEqual(VoxtralVoice.casualFemale.displayName, "Casual Female")
        XCTAssertEqual(VoxtralVoice.neutralMale.displayName, "Neutral Male")
        XCTAssertEqual(VoxtralVoice.frFemale.displayName, "Fr Female")
        XCTAssertEqual(VoxtralVoice.arMale.displayName, "Ar Male")
    }

    func testAllDisplayNamesAreNonEmpty() {
        for voice in VoxtralVoice.allCases {
            XCTAssertFalse(voice.displayName.isEmpty, "\(voice) should have a display name")
        }
    }

    // MARK: - File Name Tests

    func testEmbeddingFileName() {
        XCTAssertEqual(VoxtralVoice.neutralFemale.embeddingFileName, "voice_embedding/neutral_female.pt")
        XCTAssertEqual(VoxtralVoice.frMale.embeddingFileName, "voice_embedding/fr_male.pt")
    }

    func testSafetensorsFileName() {
        XCTAssertEqual(VoxtralVoice.neutralFemale.safetensorsFileName, "voice_embedding/neutral_female.safetensors")
        XCTAssertEqual(VoxtralVoice.frMale.safetensorsFileName, "voice_embedding/fr_male.safetensors")
    }

    func testAllFileNamesEndWithCorrectExtension() {
        for voice in VoxtralVoice.allCases {
            XCTAssertTrue(voice.embeddingFileName.hasSuffix(".pt"))
            XCTAssertTrue(voice.safetensorsFileName.hasSuffix(".safetensors"))
            XCTAssertTrue(voice.embeddingFileName.hasPrefix("voice_embedding/"))
            XCTAssertTrue(voice.safetensorsFileName.hasPrefix("voice_embedding/"))
        }
    }

    // MARK: - Language Tests

    func testEnglishVoiceLanguages() {
        XCTAssertEqual(VoxtralVoice.casualFemale.language, "en")
        XCTAssertEqual(VoxtralVoice.casualMale.language, "en")
        XCTAssertEqual(VoxtralVoice.cheerfulFemale.language, "en")
        XCTAssertEqual(VoxtralVoice.neutralFemale.language, "en")
        XCTAssertEqual(VoxtralVoice.neutralMale.language, "en")
    }

    func testFrenchVoiceLanguages() {
        XCTAssertEqual(VoxtralVoice.frMale.language, "fr")
        XCTAssertEqual(VoxtralVoice.frFemale.language, "fr")
    }

    func testSpanishVoiceLanguages() {
        XCTAssertEqual(VoxtralVoice.esMale.language, "es")
        XCTAssertEqual(VoxtralVoice.esFemale.language, "es")
    }

    func testGermanVoiceLanguages() {
        XCTAssertEqual(VoxtralVoice.deMale.language, "de")
        XCTAssertEqual(VoxtralVoice.deFemale.language, "de")
    }

    func testItalianVoiceLanguages() {
        XCTAssertEqual(VoxtralVoice.itMale.language, "it")
        XCTAssertEqual(VoxtralVoice.itFemale.language, "it")
    }

    func testPortugueseVoiceLanguages() {
        XCTAssertEqual(VoxtralVoice.ptMale.language, "pt")
        XCTAssertEqual(VoxtralVoice.ptFemale.language, "pt")
    }

    func testDutchVoiceLanguages() {
        XCTAssertEqual(VoxtralVoice.nlMale.language, "nl")
        XCTAssertEqual(VoxtralVoice.nlFemale.language, "nl")
    }

    func testArabicVoiceLanguage() {
        XCTAssertEqual(VoxtralVoice.arMale.language, "ar")
    }

    func testHindiVoiceLanguages() {
        XCTAssertEqual(VoxtralVoice.hiMale.language, "hi")
        XCTAssertEqual(VoxtralVoice.hiFemale.language, "hi")
    }

    func testAllVoicesHaveLanguage() {
        for voice in VoxtralVoice.allCases {
            XCTAssertNotNil(voice.language, "\(voice) should have a language")
        }
    }

    func testSupportedLanguages() {
        let languages = Set(VoxtralVoice.allCases.compactMap { $0.language })
        let expected: Set<String> = ["en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi"]
        XCTAssertEqual(languages, expected)
    }

    // MARK: - Voice Gender Balance

    func testEnglishVoicesHaveBothGenders() {
        let enVoices = VoxtralVoice.allCases.filter { $0.language == "en" }
        let hasFemale = enVoices.contains { $0.rawValue.contains("female") }
        let hasMale = enVoices.contains { $0.rawValue.contains("male") && !$0.rawValue.contains("female") }
        XCTAssertTrue(hasFemale)
        XCTAssertTrue(hasMale)
    }

    // MARK: - VoxtralVoicePresetManager Tests

    func testManagerInitWithDefaultCache() {
        let manager = VoxtralVoicePresetManager()
        XCTAssertTrue(manager.cacheDirectory.path.contains(".voxtral/voices"))
    }

    func testManagerInitWithCustomCache() {
        let customDir = URL(fileURLWithPath: "/tmp/test_voices")
        let manager = VoxtralVoicePresetManager(cacheDirectory: customDir)
        XCTAssertEqual(manager.cacheDirectory, customDir)
    }

    func testManagerDefaultRepoId() {
        let manager = VoxtralVoicePresetManager()
        XCTAssertEqual(manager.modelRepoId, "mistralai/Voxtral-4B-TTS-2603")
    }

    func testManagerCustomRepoId() {
        let manager = VoxtralVoicePresetManager(modelRepoId: "custom/repo")
        XCTAssertEqual(manager.modelRepoId, "custom/repo")
    }

    func testVoiceNotAvailableInEmptyDirectory() {
        let emptyDir = FileManager.default.temporaryDirectory.appendingPathComponent("empty_voices_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: emptyDir) }

        let manager = VoxtralVoicePresetManager()
        XCTAssertFalse(manager.isVoiceAvailable(.neutralFemale, in: emptyDir))
    }

    func testAvailableVoicesInEmptyDirectory() {
        let emptyDir = FileManager.default.temporaryDirectory.appendingPathComponent("empty_voices_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: emptyDir) }

        let manager = VoxtralVoicePresetManager()
        let voices = manager.availableVoices(in: emptyDir)
        XCTAssertTrue(voices.isEmpty)
    }

    func testLoadVoiceFromNonExistentDirectory() {
        let manager = VoxtralVoicePresetManager()
        let fakeDir = URL(fileURLWithPath: "/tmp/nonexistent_model_dir")

        XCTAssertThrowsError(try manager.loadVoiceEmbedding(voice: .neutralFemale, from: fakeDir)) { error in
            guard case VoxtralTTSError.voiceNotFound = error else {
                XCTFail("Expected voiceNotFound error, got \(error)")
                return
            }
        }
    }

    // MARK: - Sendable Conformance

    func testVoiceIsSendable() {
        let expectation = expectation(description: "Sendable check")
        let voice: VoxtralVoice = .neutralFemale
        Task {
            let _ = voice.rawValue
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
    }
}
