/**
 * VoxtralTTSSanitizationTests - Unit tests for TTS text sanitization
 */

import XCTest
@testable import VoxtralCore

final class VoxtralTTSSanitizationTests: XCTestCase {

    // MARK: - Terminal Punctuation

    func testAddsTerminalPeriod() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Hello world")
        XCTAssertTrue(result.hasSuffix("."))
    }

    func testPreservesExistingPeriod() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Hello world.")
        XCTAssertEqual(result, "Hello world.")
    }

    func testPreservesExclamationMark() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Hello world!")
        XCTAssertEqual(result, "Hello world!")
    }

    func testPreservesQuestionMark() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Hello world?")
        XCTAssertEqual(result, "Hello world?")
    }

    func testEmptyTextBecomesPeriod() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("")
        XCTAssertEqual(result, ".")
    }

    // MARK: - ALL-CAPS Conversion

    func testAllCapsWordConversion() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("FORGE YOUR IDEA.")
        XCTAssertEqual(result, "Forge Your Idea.")
    }

    func testSingleCharCapsNotConverted() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("I am OK.")
        XCTAssertTrue(result.contains("I"))
    }

    func testMixedCaseNotConverted() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Hello World.")
        XCTAssertEqual(result, "Hello World.")
    }

    // MARK: - Symbol Verbalization

    func testAmpersandVerbalized() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Library & Export.")
        XCTAssertTrue(result.contains(" and "))
        XCTAssertFalse(result.contains("&"))
    }

    func testPlusVerbalized() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("1300+ tests.")
        XCTAssertTrue(result.contains(" plus "))
    }

    func testEqualsVerbalized() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("ratio = 16:9.")
        XCTAssertTrue(result.contains(" equals "))
    }

    func testSlashVerbalized() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("sRGB / Display P3.")
        XCTAssertTrue(result.contains(" or "))
    }

    // MARK: - Dash Handling

    func testEmDashBecomesComma() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Forge your idea \u{2014} completely redesigned.")
        XCTAssertTrue(result.contains(","))
        XCTAssertFalse(result.contains("\u{2014}"))
    }

    func testStandaloneHyphenBecomesComma() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Forge your idea - completely redesigned.")
        XCTAssertTrue(result.contains(","))
    }

    func testInWordHyphenPreserved() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("auto-saved projects.")
        XCTAssertTrue(result.contains("auto-saved"))
    }

    // MARK: - Repeated Punctuation

    func testRepeatedExclamationCollapsed() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Amazing!!!")
        XCTAssertEqual(result, "Amazing!")
    }

    func testRepeatedPeriodsCollapsed() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Wait....")
        XCTAssertTrue(result == "Wait." || result == "Wait...")
    }

    // MARK: - Whitespace Normalization

    func testMultipleSpacesCollapsed() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Hello    world.")
        XCTAssertEqual(result, "Hello world.")
    }

    func testLeadingTrailingWhitespaceTrimmed() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("  Hello world.  ")
        XCTAssertEqual(result, "Hello world.")
    }

    // MARK: - Phase 1: Structural Transformations

    func testParagraphBreaksBecomeSentenceBoundary() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("First paragraph.\n\nSecond paragraph.")
        XCTAssertEqual(result, "First paragraph. Second paragraph.")
    }

    func testParagraphBreakAddsPeriodIfMissing() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("First paragraph\n\nSecond paragraph.")
        XCTAssertTrue(result.contains("First paragraph."))
        XCTAssertTrue(result.contains("Second paragraph."))
    }

    func testSingleNewlineBecomesSpace() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("Line one.\nLine two.")
        XCTAssertEqual(result, "Line one. Line two.")
    }

    func testSectionHeadersConverted() {
        let input = "Some text.\n\nSECTION HEADER\nContent here."
        let result = VoxtralTTSModel.sanitizeTextForTTS(input)
        // Header should be capitalized and have period
        XCTAssertTrue(result.contains("Section Header."))
        XCTAssertTrue(result.contains("Content here."))
    }

    func testBulletPointsConvertedToSentences() {
        let input = "Features:\n- Fast generation.\n- High quality.\n- Easy to use."
        let result = VoxtralTTSModel.sanitizeTextForTTS(input)
        XCTAssertTrue(result.contains("Fast generation."))
        XCTAssertTrue(result.contains("High quality."))
        XCTAssertTrue(result.contains("Easy to use."))
        XCTAssertFalse(result.contains("-"))
    }

    func testBulletPointsAddPeriodIfMissing() {
        let input = "List:\n- Item one\n- Item two"
        let result = VoxtralTTSModel.sanitizeTextForTTS(input)
        XCTAssertTrue(result.contains("Item one."))
        XCTAssertTrue(result.contains("Item two."))
    }

    func testMarkdownHeadersStripped() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("# Main Title\nContent.")
        XCTAssertFalse(result.contains("#"))
        XCTAssertTrue(result.contains("Main Title"))
    }

    func testMarkdownBoldStripped() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("This is **bold** text.")
        XCTAssertFalse(result.contains("**"))
        XCTAssertTrue(result.contains("bold"))
    }

    func testBlockquoteStripped() {
        let result = VoxtralTTSModel.sanitizeTextForTTS("> Quoted text.")
        XCTAssertFalse(result.hasPrefix(">"))
        XCTAssertTrue(result.contains("Quoted text."))
    }

    // MARK: - Plain Text Passthrough

    func testPlainTextUnchanged() {
        let input = "Hello world."
        let result = VoxtralTTSModel.sanitizeTextForTTS(input)
        XCTAssertEqual(result, "Hello world.")
    }

    func testPlainSentenceWithComma() {
        let input = "Hello, world."
        let result = VoxtralTTSModel.sanitizeTextForTTS(input)
        XCTAssertEqual(result, "Hello, world.")
    }

    // MARK: - Full Integration

    func testFluxforgeShortText() {
        let input = "Fluxforge Studio transforme votre Mac en un studio de création IA complet."
        let result = VoxtralTTSModel.sanitizeTextForTTS(input)
        // "IA" (2 uppercase chars) gets capitalized to "Ia"
        XCTAssertEqual(result, "Fluxforge Studio transforme votre Mac en un studio de création Ia complet.")
    }

    func testFluxforgeStructuredText() {
        let input = """
        Introduction text.

        FORGE TON IDÉE
        Un atelier créatif complet.

        GÉNÉRATION VIDÉO
        Créez des vidéos.
        - Distilled : rapide.
        - Dev : haute qualité.
        """
        let result = VoxtralTTSModel.sanitizeTextForTTS(input)

        // Should contain capitalized headers
        XCTAssertTrue(result.contains("Forge Ton Idée."))
        XCTAssertTrue(result.contains("Génération Vidéo."))
        // Should contain bullet items as sentences
        XCTAssertTrue(result.contains("Distilled : rapide."))
        XCTAssertTrue(result.contains("Dev : haute qualité."))
        // Should not contain bullet markers
        XCTAssertFalse(result.contains("- "))
        // Should be single line
        XCTAssertFalse(result.contains("\n"))
    }
}
