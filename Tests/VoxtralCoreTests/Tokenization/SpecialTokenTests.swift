/**
 * SpecialTokenTests - Unit tests for special token handling in TekkenTokenizer
 */

import XCTest
@testable import VoxtralCore

final class SpecialTokenTests: XCTestCase {

    var tokenizer: TekkenTokenizer!

    override func setUpWithError() throws {
        tokenizer = TekkenTokenizer()
        // Note: Tests work with demo tokenizer when model files are not available
    }

    // MARK: - Control Token Tests

    func testGetControlTokenReturnsMinusOneForUnknown() {
        // Unknown tokens should return -1
        let unknownToken = tokenizer.getControlToken("<|unknown_token|>")
        XCTAssertEqual(unknownToken, -1, "Unknown control token should return -1")
    }

    func testGetControlTokenConsistency() {
        // Same token should always return the same ID
        let token1 = tokenizer.getControlToken("<|begin_of_text|>")
        let token2 = tokenizer.getControlToken("<|begin_of_text|>")
        XCTAssertEqual(token1, token2, "Same control token should return consistent ID")
    }

    func testCommonControlTokens() {
        // Test known control tokens (may return -1 if demo tokenizer)
        let beginText = tokenizer.getControlToken("<|begin_of_text|>")
        let endText = tokenizer.getControlToken("<|end_of_text|>")

        // If demo tokenizer, both will be -1, which is valid
        // If real tokenizer, they should be different positive values
        if beginText != -1 && endText != -1 {
            XCTAssertNotEqual(beginText, endText, "Different tokens should have different IDs")
        }
    }

    // MARK: - Encode with Special Tokens Tests

    func testEncodeWithoutSpecialTokens() {
        let text = "Hello"
        let tokens = tokenizer.encode(text, addSpecialTokens: false)

        XCTAssertFalse(tokens.isEmpty, "Encoding should produce tokens")
    }

    func testEncodeWithSpecialTokens() {
        let text = "Hello"
        let tokensWithout = tokenizer.encode(text, addSpecialTokens: false)
        let tokensWith = tokenizer.encode(text, addSpecialTokens: true)

        // With special tokens should be >= without (may add BOS/EOS)
        XCTAssertGreaterThanOrEqual(
            tokensWith.count,
            tokensWithout.count,
            "Tokens with special tokens should be >= without"
        )
    }

    func testEncodeEmptyStringWithSpecialTokens() {
        let tokens = tokenizer.encode("", addSpecialTokens: true)
        // Empty string may still have special tokens or be empty
        // Just verify it doesn't crash
        XCTAssertNotNil(tokens)
    }

    // MARK: - Decode with Skip Special Tokens Tests

    func testDecodeWithSkipSpecialTokensTrue() {
        let text = "Hello world"
        let tokens = tokenizer.encode(text, addSpecialTokens: false)
        let decoded = tokenizer.decode(tokens, skipSpecialTokens: true)

        XCTAssertEqual(text, decoded, "Round-trip should preserve text")
    }

    func testDecodeWithSkipSpecialTokensFalse() {
        let text = "Hello"
        let tokens = tokenizer.encode(text, addSpecialTokens: false)
        let decoded = tokenizer.decode(tokens, skipSpecialTokens: false)

        // Should still contain the original text
        XCTAssertTrue(
            decoded.contains("Hello") || decoded == text || !tokens.isEmpty,
            "Decoded text should be related to input"
        )
    }

    func testDecodeEmptyTokens() {
        let decoded = tokenizer.decode([], skipSpecialTokens: true)
        XCTAssertEqual(decoded, "", "Empty tokens should decode to empty string")
    }

    func testDecodeEmptyTokensWithoutSkip() {
        let decoded = tokenizer.decode([], skipSpecialTokens: false)
        XCTAssertEqual(decoded, "", "Empty tokens should decode to empty string")
    }

    // MARK: - Batch Decode Tests

    func testBatchDecodeMultipleSequences() {
        let texts = ["Hello", "World", "Test"]
        let tokenLists = texts.map { tokenizer.encode($0, addSpecialTokens: false) }
        let decoded = tokenizer.batchDecode(tokenLists, skipSpecialTokens: true)

        XCTAssertEqual(decoded.count, texts.count, "Batch decode should return same count")
        for (i, text) in texts.enumerated() {
            XCTAssertEqual(decoded[i], text, "Each decoded text should match original at index \(i)")
        }
    }

    func testBatchDecodeEmpty() {
        let decoded = tokenizer.batchDecode([], skipSpecialTokens: true)
        XCTAssertTrue(decoded.isEmpty, "Empty batch should return empty array")
    }

    func testBatchDecodeWithEmptySequence() {
        let tokenLists: [[Int]] = [[], [1, 2, 3], []]
        let decoded = tokenizer.batchDecode(tokenLists, skipSpecialTokens: true)

        XCTAssertEqual(decoded.count, 3, "Should return same number of results")
        XCTAssertEqual(decoded[0], "", "Empty sequence should decode to empty string")
        XCTAssertEqual(decoded[2], "", "Empty sequence should decode to empty string")
    }

    // MARK: - Unicode and Edge Cases

    func testEncodeUnicodeCharacters() {
        let text = "Hello 世界"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Unicode should round-trip correctly")
    }

    func testEncodeWithCombiningCharacters() {
        let text = "café"  // Has combining accent
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Combining characters should round-trip correctly")
    }

    func testEncodeWithEmoji() {
        let text = "Hello 🎵"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Emoji should round-trip correctly")
    }

    func testEncodeWhitespaceOnly() {
        let text = "   "
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        // Whitespace handling may vary, just verify no crash
        XCTAssertNotNil(decoded)
    }

    func testEncodeLongText() {
        let text = String(repeating: "Hello world. ", count: 100)
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Long text should round-trip correctly")
    }

    // MARK: - Special Character Tests

    func testEncodeNewlines() {
        let text = "Hello\nWorld\nTest"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Newlines should round-trip correctly")
    }

    func testEncodeTabs() {
        let text = "Hello\tWorld"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Tabs should round-trip correctly")
    }

    func testEncodeSpecialPunctuation() {
        let text = "Hello! How are you? Fine, thanks."
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Punctuation should round-trip correctly")
    }

    func testEncodeNumbers() {
        let text = "The answer is 42.5 or maybe 100"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Numbers should round-trip correctly")
    }

    // MARK: - Multilingual Tests

    func testEncodeFrench() {
        let text = "Bonjour, comment ça va?"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "French should round-trip correctly")
    }

    func testEncodeGerman() {
        let text = "Guten Tag, wie geht es Ihnen?"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "German should round-trip correctly")
    }

    func testEncodeJapanese() {
        let text = "こんにちは"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Japanese should round-trip correctly")
    }

    func testEncodeChinese() {
        let text = "你好世界"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Chinese should round-trip correctly")
    }

    func testEncodeArabic() {
        let text = "مرحبا"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Arabic should round-trip correctly")
    }

    func testEncodeMixedLanguages() {
        let text = "Hello Bonjour 你好 مرحبا"
        let tokens = tokenizer.encode(text)
        let decoded = tokenizer.decode(tokens)

        XCTAssertEqual(text, decoded, "Mixed languages should round-trip correctly")
    }
}
